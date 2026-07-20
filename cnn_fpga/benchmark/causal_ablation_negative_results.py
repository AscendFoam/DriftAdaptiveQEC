"""T5.4.3 causal mechanism ablations and negative-result ledger.

The six requested switches do not coexist in one validated end-to-end model.
This module therefore executes or reconstructs each intervention in its native,
matched evidence lane and forbids a cross-lane score.  History, regime state,
run length, parameter update, and fallback use already formal same-trace parent
campaigns.  The retired CNN-residual branch is re-inferred on its preserved
held-out test split with the residual set exactly to zero; that lane is limited
to residual-parameter MSE and is never promoted to logical or control gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import algorithm_success_falsification as branch_parent
from cnn_fpga.benchmark import memory_specific_ablation as history_parent
from cnn_fpga.benchmark import regime_hmm_baseline as regime_parent
from cnn_fpga.benchmark import run_length_fsm_baseline as event_parent
from cnn_fpga.benchmark import uncertainty_gated_fallback as fallback_parent
from cnn_fpga.benchmark.continuous_adaptive_map import _mean_interval
from cnn_fpga.model.tiny_cnn import predict_from_artifact


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.4.3"
SCHEMA_VERSION = "t5.4.3-causal-ablation-negative-results-v1"
PROTOCOL_ID = "NATIVE-LANE-CAUSAL-ABLATION-NONMIXING-V1"
DEFAULT_ARTIFACT = Path("docs/t5_4_3_causal_ablation_negative_results.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_4_3_causal_ablation_negative_results_source_data.csv")

MECHANISMS = (
    "history",
    "cnn_residual",
    "regime_state",
    "run_length",
    "parameter_update",
    "fallback",
)

PARENT_ARTIFACTS: dict[str, Path] = {
    "T3.2.11": Path("docs/t3_2_11_memory_specific_ablation_validation.json"),
    "T3.2.6": Path("docs/t3_2_6_regime_hmm_validation.json"),
    "T3.2.5": Path("docs/t3_2_5_run_length_fsm_validation.json"),
    "T5.1.4": Path("docs/t5_1_4_algorithm_branch_verdict.json"),
    "T5.4.2": Path("docs/t5_4_2_uncertainty_gated_fallback.json"),
}

PARENT_SOURCE_FILES: dict[str, Path] = {
    "T3.2.11": Path("docs/t3_2_11_memory_specific_ablation_source_data.csv"),
    "T3.2.6": Path("docs/t3_2_6_regime_hmm_source_data.csv"),
    "T3.2.5": Path("docs/t3_2_5_run_length_fsm_source_data.csv"),
    "T5.4.2": Path("docs/t5_4_2_uncertainty_gated_fallback_source_data.csv"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/causal_ablation_negative_results.py"),
    Path("cnn_fpga/benchmark/memory_specific_ablation.py"),
    Path("cnn_fpga/benchmark/regime_hmm_baseline.py"),
    Path("cnn_fpga/benchmark/run_length_fsm_baseline.py"),
    Path("cnn_fpga/benchmark/uncertainty_gated_fallback.py"),
    Path("cnn_fpga/model/tiny_cnn.py"),
)

CNN_ASSETS: dict[str, Path] = {
    "manifest": Path("artifacts/datasets/runtime_b_residual_v1/manifest.json"),
    "test_split": Path("artifacts/datasets/runtime_b_residual_v1/test.npz"),
    "model": Path(
        "artifacts/models/runtime_b_residual_v1/"
        "tiny_cnn_20260401_083648_2fc740424c0d.npz"
    ),
    "evaluation_report": Path(
        "artifacts/reports/runtime_b_residual_v1/eval_test_20260401_083649.json"
    ),
}


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
    if payload.get("status") == "PASS" or payload.get("passed") is True:
        gate = payload.get("gate_summary")
        if isinstance(gate, Mapping):
            failed = gate.get("failed", gate.get("failed_names", 0))
            if isinstance(failed, Sequence) and not isinstance(
                failed, (str, bytes)
            ):
                if len(failed) != 0:
                    return False
            elif int(failed or 0) != 0:
                return False
        return True
    return False


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


def _history_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    metric = "logical_z_effective_lifetime_cycles"
    split_rows: dict[str, Any] = {}
    for split in ("primary", "confirmation"):
        summary = parent["summary"][split]
        active = [float(value) for value in summary["full_history"][metric]["values"]]
        off = [
            float(value)
            for value in summary["frozen_parent_last_outcome_only"][metric]["values"]
        ]
        capacity = [
            float(value)
            for value in summary["retrained_exact_budget_last_outcome"][metric]["values"]
        ]
        direct = parent["paired_bootstrap_full_minus_ablation"][split]
        interval = direct["full_minus_frozen_parent_last_outcome_only"][metric]
        capacity_interval = direct[
            "full_minus_retrained_exact_budget_last_outcome"
        ][metric]
        agent_rows = [
            {
                "agent_index": index,
                "active_full_history": active_value,
                "off_frozen_latest_only": off_value,
                "benefit_active_minus_off": active_value - off_value,
                "capacity_control_retrained_latest_only": capacity_value,
                "benefit_active_minus_capacity_control": active_value
                - capacity_value,
            }
            for index, (active_value, off_value, capacity_value) in enumerate(
                zip(active, off, capacity, strict=True)
            )
        ]
        split_rows[split] = {
            "cutoff": int(
                parent["parent_contract"][
                    "primary_cutoff" if split == "primary" else "confirmation_cutoff"
                ]
            ),
            "agent_rows": agent_rows,
            "active_mean": float(np.mean(active)),
            "off_mean": float(np.mean(off)),
            "benefit_interval": interval,
            "capacity_control_mean": float(np.mean(capacity)),
            "capacity_control_benefit_interval": capacity_interval,
        }
    return {
        "mechanism": "history",
        "native_lane": "finite_cutoff_two_level_nmf_control",
        "parent_task": "T3.2.11",
        "active": "frozen_parent_gru_full_observed_prefix",
        "off_intervention": "same_frozen_weights_zero_state_latest_observed_token_only",
        "capacity_control": "independently_retrained_exact_budget_latest_outcome_fnn",
        "metric": metric,
        "metric_direction": "higher_is_better",
        "splits": split_rows,
        "intervention_changes_actions": bool(
            parent["action_intervention_audit"]["every_intervention_changes_actions"]
        ),
        "result": "CROSS_CUTOFF_REVERSAL_NOT_SUPPORTED",
        "claim_decision": "DOWNGRADE_MEMORY_MECHANISM_NOT_SUPPORTED",
        "scope": "matched finite-cutoff two-level ten-cycle controller only",
    }


def _cnn_lane() -> dict[str, Any]:
    manifest = json.loads(_repo_path(CNN_ASSETS["manifest"]).read_text(encoding="utf-8"))
    evaluation = json.loads(
        _repo_path(CNN_ASSETS["evaluation_report"]).read_text(encoding="utf-8")
    )
    with np.load(_repo_path(CNN_ASSETS["test_split"]), allow_pickle=True) as data:
        histograms = np.asarray(data["histograms"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float64)
        scenarios = np.asarray(data["scenario_names"]).astype(str)
        window_ids = np.asarray(data["window_ids"], dtype=np.int64)
        label_names = tuple(str(value) for value in data["label_names"])
    active = np.asarray(
        predict_from_artifact(_repo_path(CNN_ASSETS["model"]), histograms),
        dtype=np.float64,
    )
    off = np.zeros_like(active)
    active_per_sample = np.mean(np.square(active - labels), axis=1)
    off_per_sample = np.mean(np.square(off - labels), axis=1)
    samples = [
        {
            "sample_id": int(index),
            "scenario_id": str(scenarios[index]),
            "window_id": int(window_ids[index]),
            "target_delta_b": labels[index].tolist(),
            "active_predicted_delta_b": active[index].tolist(),
            "off_predicted_delta_b": off[index].tolist(),
            "active_squared_error": float(active_per_sample[index]),
            "off_squared_error": float(off_per_sample[index]),
            "benefit_off_minus_active_mse": float(
                off_per_sample[index] - active_per_sample[index]
            ),
        }
        for index in range(labels.shape[0])
    ]
    scenario_rows = []
    for scenario in sorted(set(scenarios)):
        selected = scenarios == scenario
        active_mse = float(np.mean(active_per_sample[selected]))
        off_mse = float(np.mean(off_per_sample[selected]))
        scenario_rows.append(
            {
                "scenario_id": str(scenario),
                "samples": int(np.count_nonzero(selected)),
                "active_mse": active_mse,
                "off_mse": off_mse,
                "benefit_off_minus_active_mse": off_mse - active_mse,
            }
        )
    active_mse = float(np.mean(active_per_sample))
    off_mse = float(np.mean(off_per_sample))
    return {
        "mechanism": "cnn_residual",
        "native_lane": "retired_legacy_residual_b_held_out_parameter_prediction",
        "parent_task": "legacy_artifact_with_T5.1.4_revocation",
        "active": "preserved_tiny_cnn_float_predicted_delta_b",
        "off_intervention": "same_test_inputs_exact_zero_delta_b",
        "metric": "residual_b_mean_squared_error",
        "metric_direction": "lower_is_better",
        "label_names": list(label_names),
        "manifest_label_semantics": manifest["label_semantics"],
        "samples": samples,
        "scenario_rows": scenario_rows,
        "aggregate": {
            "samples": int(labels.shape[0]),
            "active_mse": active_mse,
            "off_mse": off_mse,
            "benefit_off_minus_active_mse": off_mse - active_mse,
            "active_mae": float(np.mean(np.abs(active - labels))),
            "off_mae": float(np.mean(np.abs(labels))),
            "preserved_evaluation_report_mse": float(evaluation["metrics"]["mse"]),
        },
        "uncertainty_status": "NO_INDEPENDENT_SEED_CLUSTER_CI_SINGLE_LEGACY_TEST_SPLIT",
        "result": "PARAMETER_MSE_IMPROVES_BUT_ACTIVE_BRANCH_REVOKED",
        "claim_decision": "RETAIN_METHOD_DETAIL_ONLY_REMOVE_PERFORMANCE_CLAIM",
        "scope": "held-out residual-parameter prediction only; not LER or control gain",
    }


def _regime_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    seed_rows = []
    for row in parent["evaluation"]["per_seed"]:
        active = row["causal_hmm"]
        off = row["memoryless_emission"]
        seed_rows.append(
            {
                "seed": int(row["evaluation_seed"]),
                "active_hmm_nll": float(active["negative_log_likelihood"]),
                "off_memoryless_nll": float(off["negative_log_likelihood"]),
                "benefit_off_minus_active_nll": float(
                    off["negative_log_likelihood"]
                    - active["negative_log_likelihood"]
                ),
                "active_detection_delay_windows": float(
                    active["mean_transition_detection_delay_windows"]
                ),
                "off_detection_delay_windows": float(
                    off["mean_transition_detection_delay_windows"]
                ),
                "active_minus_off_delay_cost": float(
                    active["mean_transition_detection_delay_windows"]
                    - off["mean_transition_detection_delay_windows"]
                ),
            }
        )
    return {
        "mechanism": "regime_state",
        "native_lane": "synthetic_four_regime_observed_window_estimation",
        "parent_task": "T3.2.6",
        "active": "causal_hmm_with_temporal_regime_posterior",
        "off_intervention": "same_fitted_emission_likelihood_without_temporal_state",
        "metric": "negative_log_likelihood",
        "metric_direction": "lower_is_better",
        "seed_rows": seed_rows,
        "benefit_interval": _mean_interval(
            [row["benefit_off_minus_active_nll"] for row in seed_rows], 0.95
        ),
        "delay_cost_interval": _mean_interval(
            [row["active_minus_off_delay_cost"] for row in seed_rows], 0.95
        ),
        "result": "PROPER_SCORE_BENEFIT_WITH_DETECTION_DELAY_COST",
        "claim_decision": "RETAIN_ESTIMATOR_PROPER_SCORE_ONLY",
        "scope": "synthetic regime estimation; no logical/control gain",
    }


def _read_event_rows() -> list[dict[str, str]]:
    path = _repo_path(PARENT_SOURCE_FILES["T3.2.5"])
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _event_lanes(parent: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _read_event_rows()
    cells = []
    for row in raw:
        run_cost = float(row["run_length_fsm_event_plus_write_cost"])
        memoryless_cost = float(row["memoryless_event_event_plus_write_cost"])
        static_cost = float(row["static_safe_normal_event_plus_write_cost"])
        cells.append(
            {
                "scenario_id": row["scenario_id"],
                "seed": int(row["base_evaluation_seed"]),
                "trace_sha256": row["trace_sha256"],
                "cycles": int(row["cycles"]),
                "run_length_active_cost": run_cost,
                "run_length_off_memoryless_cost": memoryless_cost,
                "run_length_benefit_off_minus_active": memoryless_cost - run_cost,
                "parameter_update_off_static_cost": static_cost,
                "parameter_update_benefit_off_minus_active": static_cost - run_cost,
                "run_length_bank_writes": int(row["run_length_fsm_bank_writes"]),
                "off_static_bank_writes": int(row["static_safe_normal_bank_writes"]),
            }
        )
    seeds = sorted({row["seed"] for row in cells})
    seed_rows = []
    for seed in seeds:
        selected = [row for row in cells if row["seed"] == seed]
        seed_rows.append(
            {
                "seed": seed,
                "scenario_cells": len(selected),
                "run_length_benefit_off_minus_active": float(
                    np.mean(
                        [row["run_length_benefit_off_minus_active"] for row in selected]
                    )
                ),
                "parameter_update_benefit_off_minus_active": float(
                    np.mean(
                        [
                            row["parameter_update_benefit_off_minus_active"]
                            for row in selected
                        ]
                    )
                ),
            }
        )
    scenario_rows = []
    for scenario in sorted({row["scenario_id"] for row in cells}):
        selected = [row for row in cells if row["scenario_id"] == scenario]
        scenario_rows.append(
            {
                "scenario_id": scenario,
                "cells": len(selected),
                "run_length_benefit_interval": _mean_interval(
                    [row["run_length_benefit_off_minus_active"] for row in selected],
                    0.95,
                ),
                "parameter_update_benefit_interval": _mean_interval(
                    [
                        row["parameter_update_benefit_off_minus_active"]
                        for row in selected
                    ],
                    0.95,
                ),
            }
        )
    run_interval = _mean_interval(
        [row["run_length_benefit_off_minus_active"] for row in seed_rows], 0.95
    )
    update_interval = _mean_interval(
        [row["parameter_update_benefit_off_minus_active"] for row in seed_rows],
        0.95,
    )
    shared = {
        "native_lane": "observed_event_controller_same_trace_software_cost",
        "parent_task": "T3.2.5",
        "metric": "event_plus_write_cost",
        "metric_direction": "lower_is_better",
        "cells": cells,
        "seed_rows": seed_rows,
        "scenario_rows": scenario_rows,
        "evaluation_cycles": int(parent["aggregate"]["evaluation_cycles"]),
    }
    run_lane = {
        "mechanism": "run_length",
        **shared,
        "active": "nondegenerate_three_bit_run_length_fsm_with_atomic_updates",
        "off_intervention": "same_observation_current_event_memoryless_controller",
        "benefit_interval": run_interval,
        "result": "NEGATIVE_MEMORYLESS_CONTROLLER_IS_BETTER",
        "claim_decision": "DOWNGRADE_RUN_LENGTH_PERFORMANCE_ACTIVE_IS_WORSE",
        "scope": "software event cost; not logical error rate",
    }
    update_lane = {
        "mechanism": "parameter_update",
        **shared,
        "active": "run_length_decisions_committed_through_real_double_param_bank",
        "off_intervention": "hold_last_known_good_normal_image_while_retaining_local_health_fallback",
        "benefit_interval": update_interval,
        "result": "UPDATES_HELP_COMPONENT_EVENT_COST",
        "claim_decision": "RETAIN_COMPONENT_EVENT_COST_ONLY",
        "scope": "software event actuation; not decoder or physical-memory gain",
    }
    return run_lane, update_lane


def _fallback_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    confirmation = parent["confirmation_ood"]
    nominal = parent["confirmation_nominal"]
    scenarios = [
        {
            "scenario_id": scenario_id,
            "benefit_interval": summary["metrics"][
                "absolute_catastrophic_reduction"
            ],
            "active_gated_failure_rate": summary["metrics"]["gated_failure_rate"],
            "off_primary_failure_rate": summary["metrics"]["primary_failure_rate"],
            "fallback_rate": summary["metrics"]["fallback_rate"],
        }
        for scenario_id, summary in confirmation["scenario_summaries"].items()
    ]
    return {
        "mechanism": "fallback",
        "native_lane": "fresh_confirmatory_ood_syndrome_decision",
        "parent_task": "T5.4.2",
        "active": "observed_uncertainty_gate_selects_frozen_static_map",
        "off_intervention": "always_use_frozen_ewma_primary_on_same_sample",
        "metric": "logical_class_failure_rate",
        "metric_direction": "lower_is_better",
        "seed_rows": confirmation["seed_rows"],
        "scenario_rows": scenarios,
        "benefit_interval": confirmation["metrics"][
            "absolute_catastrophic_reduction"
        ],
        "sample_accounting": confirmation["sample_accounting"],
        "nominal_benefit_interval": nominal["metrics"][
            "absolute_catastrophic_reduction"
        ],
        "result": "AGGREGATE_POSITIVE_BUT_SCENARIO_NONUNIVERSAL",
        "claim_decision": "RETAIN_MIXTURE_QUALIFIED_AGGREGATE_ONLY",
        "scope": "synthetic syndrome-decision level; not physical memory or device",
    }


def _build_lanes(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    run_lane, update_lane = _event_lanes(parents["T3.2.5"])
    lanes = {
        "history": _history_lane(parents["T3.2.11"]),
        "cnn_residual": _cnn_lane(),
        "regime_state": _regime_lane(parents["T3.2.6"]),
        "run_length": run_lane,
        "parameter_update": update_lane,
        "fallback": _fallback_lane(parents["T5.4.2"]),
    }
    if tuple(lanes) != MECHANISMS:
        raise RuntimeError("mechanism order drifted")
    return lanes


def _negative_result_table(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "mechanism": mechanism,
            "native_lane": lanes[mechanism]["native_lane"],
            "metric": lanes[mechanism]["metric"],
            "result": lanes[mechanism]["result"],
            "claim_decision": lanes[mechanism]["claim_decision"],
            "scope": lanes[mechanism]["scope"],
        }
        for mechanism in MECHANISMS
    ]


def _contract_view(report: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"generated_at_utc", "contract_sha256", "source_data", "gate_summary"}
    return {key: value for key, value in report.items() if key not in excluded}


def _current_implementation_hashes() -> dict[str, str]:
    return {
        "T3.2.11": history_parent.implementation_sha256(),
        "T3.2.6": regime_parent._implementation_sha256(),
        "T3.2.5": event_parent._implementation_sha256(),
        "T5.1.4": branch_parent.implementation_sha256(),
    }


def _compute_gates(
    report: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    lanes = report["lanes"]
    history = lanes["history"]
    cnn = lanes["cnn_residual"]
    regime = lanes["regime_state"]
    run_length = lanes["run_length"]
    update = lanes["parameter_update"]
    fallback = lanes["fallback"]
    parent_impl = _current_implementation_hashes()
    return {
        "six_registered_mechanisms_have_native_interventions": tuple(lanes)
        == MECHANISMS
        and all(lanes[name]["active"] and lanes[name]["off_intervention"] for name in MECHANISMS),
        "all_parent_artifacts_are_hash_bound_and_machine_pass": all(
            row["machine_pass"] and row["sha256"] == _sha256(row["path"])
            for row in report["parent_bindings"]
        ),
        "all_parent_source_files_are_hash_bound": all(
            row["sha256"] == _sha256(row["path"])
            for row in report["parent_source_bindings"]
        ),
        "all_implementation_and_cnn_assets_are_hash_bound": all(
            row["sha256"] == _sha256(row["path"])
            for row in (*report["implementation_bindings"], *report["cnn_asset_bindings"])
        ),
        "parent_implementation_composites_are_current": parents["T3.2.11"][
            "implementation_sha256"
        ]
        == parent_impl["T3.2.11"]
        and parents["T3.2.6"]["implementation_sha256"] == parent_impl["T3.2.6"]
        and parents["T3.2.5"]["implementation_sha256"] == parent_impl["T3.2.5"]
        and parents["T5.1.4"]["implementation_sha256"] == parent_impl["T5.1.4"],
        "history_intervention_is_action_changing_and_cross_cutoff_reversal_retained": history[
            "intervention_changes_actions"
        ]
        and history["splits"]["primary"]["benefit_interval"]["ci95_low"] > 0.0
        and history["splits"]["confirmation"]["benefit_interval"]["ci95_high"] < 0.0
        and history["claim_decision"] == "DOWNGRADE_MEMORY_MECHANISM_NOT_SUPPORTED",
        "cnn_zero_residual_is_exact_and_reproduces_preserved_test_mse": all(
            np.asarray(row["off_predicted_delta_b"]) .tolist() == [0.0, 0.0]
            for row in cnn["samples"]
        )
        and abs(
            cnn["aggregate"]["active_mse"]
            - cnn["aggregate"]["preserved_evaluation_report_mse"]
        )
        <= 1.0e-18
        and all(row["benefit_off_minus_active_mse"] > 0.0 for row in cnn["scenario_rows"]),
        "cnn_lane_is_downgraded_to_legacy_parameter_prediction": cnn[
            "uncertainty_status"
        ]
        == "NO_INDEPENDENT_SEED_CLUSTER_CI_SINGLE_LEGACY_TEST_SPLIT"
        and cnn["claim_decision"]
        == "RETAIN_METHOD_DETAIL_ONLY_REMOVE_PERFORMANCE_CLAIM"
        and parents["T5.1.4"]["active_branch"]["strong_branch_activated"] is False,
        "regime_state_improves_proper_score_but_delay_cost_is_retained": regime[
            "benefit_interval"
        ]["ci_low"]
        > 0.0
        and regime["delay_cost_interval"]["ci_low"] > 0.0
        and len(regime["seed_rows"]) == 8,
        "run_length_negative_result_is_retained": run_length["benefit_interval"][
            "ci_high"
        ]
        < 0.0
        and run_length["claim_decision"]
        == "DOWNGRADE_RUN_LENGTH_PERFORMANCE_ACTIVE_IS_WORSE",
        "parameter_update_component_benefit_is_matched_and_bounded": update[
            "benefit_interval"
        ]["ci_low"]
        > 0.0
        and all(row["run_length_bank_writes"] > 0 for row in update["cells"])
        and all(row["off_static_bank_writes"] >= 0 for row in update["cells"]),
        "event_ablation_grid_is_complete_and_same_trace": len(run_length["cells"])
        == 32
        and len({row["trace_sha256"] for row in run_length["cells"]}) == 32
        and run_length["cells"] == update["cells"],
        "fallback_aggregate_benefit_and_scenario_harm_are_both_retained": fallback[
            "benefit_interval"
        ]["ci_low"]
        > 0.0
        and any(row["benefit_interval"]["ci_high"] < 0.0 for row in fallback["scenario_rows"])
        and fallback["nominal_benefit_interval"]["estimate"] < 0.0,
        "fallback_accounting_is_not_definitionally_positive": fallback[
            "sample_accounting"
        ]["avoided_failure_count"]
        > fallback["sample_accounting"]["induced_failure_count"]
        > 0
        and fallback["sample_accounting"]["unnecessary_fallback_count"] > 0,
        "negative_result_table_is_complete_and_claim_decisions_are_explicit": len(
            report["negative_result_table"]
        )
        == 6
        and {row["mechanism"] for row in report["negative_result_table"]}
        == set(MECHANISMS)
        and all(row["claim_decision"] for row in report["negative_result_table"]),
        "no_cross_lane_score_or_ranking_exists": report["nonmixing_contract"][
            "cross_lane_aggregate"
        ]
        is None
        and report["nonmixing_contract"]["global_ranking"] is None,
        "hidden_truth_is_offline_evaluation_only": report["causal_contract"][
            "deployment_hidden_truth_inputs"
        ]
        == []
        and report["causal_contract"]["truth_role"] == "offline_outcome_scoring_only",
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
            "all six requested mechanism-off interventions are executable and "
            "traceable in their native matched lanes; negative results and claim "
            "downgrades are preserved without a cross-lane score"
        ),
        "parent_bindings": _parent_bindings(parents),
        "parent_source_bindings": _bindings(PARENT_SOURCE_FILES, kind="parent_source"),
        "implementation_bindings": _bindings(IMPLEMENTATION_PATHS, kind="implementation"),
        "cnn_asset_bindings": _bindings(CNN_ASSETS, kind="legacy_cnn_asset"),
        "causal_contract": {
            "intervention_unit": "mechanism_off_within_native_parent_lane",
            "same_trace_rule": (
                "active and off consume the same stored observation/target unit within "
                "each lane; metrics never cross lanes"
            ),
            "deployment_hidden_truth_inputs": [],
            "truth_role": "offline_outcome_scoring_only",
            "positive_benefit_convention": (
                "active-minus-off for higher-is-better metrics and off-minus-active "
                "for lower-is-better metrics"
            ),
        },
        "nonmixing_contract": {
            "native_lanes": sorted({lane["native_lane"] for lane in lanes.values()}),
            "cross_lane_aggregate": None,
            "global_ranking": None,
            "reason": (
                "controller lifetime, parameter MSE, regime NLL, event cost, and "
                "syndrome logical-class error are different estimands"
            ),
        },
        "lanes": lanes,
        "negative_result_table": _negative_result_table(lanes),
        "claim_boundary": {
            "allowed": (
                "native-lane causal mechanism-off contrasts with explicit negative "
                "results, costs, and claim downgrades"
            ),
            "forbidden": (
                "one integrated-system causal ranking, universal mechanism benefit, "
                "physical-memory LER, device safety, RTL, FPGA, board, or experiment"
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

    def append(
        row_type: str,
        mechanism: str,
        record_id: str,
        metric: str,
        value: Any,
        detail: Any,
    ) -> None:
        rows.append(
            {
                "row_type": row_type,
                "mechanism": mechanism,
                "record_id": record_id,
                "metric": metric,
                "value": value,
                "detail": "" if detail == "" else json.dumps(detail, sort_keys=True),
            }
        )

    for binding in report["parent_bindings"]:
        append("parent_binding", "provenance", binding["task_id"], "machine_pass", int(binding["machine_pass"]), binding)
    for binding in (
        *report["parent_source_bindings"],
        *report["implementation_bindings"],
        *report["cnn_asset_bindings"],
    ):
        append("file_binding", "provenance", binding["binding_id"], "sha256_bound", 1, binding)

    history = report["lanes"]["history"]
    for split, section in history["splits"].items():
        for row in section["agent_rows"]:
            append("history_agent", "history", f"{split}-a{row['agent_index']}", "benefit_active_minus_off", row["benefit_active_minus_off"], row)
        append("history_aggregate", "history", split, "benefit_active_minus_off", section["benefit_interval"]["mean_difference"], section)

    cnn = report["lanes"]["cnn_residual"]
    for row in cnn["samples"]:
        append("cnn_sample", "cnn_residual", str(row["sample_id"]), "benefit_off_minus_active_mse", row["benefit_off_minus_active_mse"], row)
    for row in cnn["scenario_rows"]:
        append("cnn_scenario", "cnn_residual", row["scenario_id"], "benefit_off_minus_active_mse", row["benefit_off_minus_active_mse"], row)
    append("cnn_aggregate", "cnn_residual", "all", "benefit_off_minus_active_mse", cnn["aggregate"]["benefit_off_minus_active_mse"], cnn["aggregate"])

    regime = report["lanes"]["regime_state"]
    for row in regime["seed_rows"]:
        append("regime_seed", "regime_state", str(row["seed"]), "benefit_off_minus_active_nll", row["benefit_off_minus_active_nll"], row)
    append("regime_aggregate", "regime_state", "all", "benefit_off_minus_active_nll", regime["benefit_interval"]["estimate"], {"benefit": regime["benefit_interval"], "delay_cost": regime["delay_cost_interval"]})

    event = report["lanes"]["run_length"]
    for row in event["cells"]:
        append("event_cell", "run_length|parameter_update", f"{row['scenario_id']}-s{row['seed']}", "run_length_benefit", row["run_length_benefit_off_minus_active"], row)
    for row in event["seed_rows"]:
        append("event_seed", "run_length|parameter_update", str(row["seed"]), "run_length_benefit", row["run_length_benefit_off_minus_active"], row)
    for row in event["scenario_rows"]:
        append("event_scenario", "run_length|parameter_update", row["scenario_id"], "run_length_benefit", row["run_length_benefit_interval"]["estimate"], row)
    for mechanism in ("run_length", "parameter_update"):
        lane = report["lanes"][mechanism]
        append("event_aggregate", mechanism, "all", "benefit", lane["benefit_interval"]["estimate"], lane["benefit_interval"])

    fallback = report["lanes"]["fallback"]
    for row in fallback["seed_rows"]:
        append("fallback_seed", "fallback", str(row["seed"]), "absolute_catastrophic_reduction", row["absolute_catastrophic_reduction"], row)
    for row in fallback["scenario_rows"]:
        append("fallback_scenario", "fallback", row["scenario_id"], "absolute_catastrophic_reduction", row["benefit_interval"]["estimate"], row)
    append("fallback_aggregate", "fallback", "ood", "absolute_catastrophic_reduction", fallback["benefit_interval"]["estimate"], fallback["benefit_interval"])
    append("fallback_aggregate", "fallback", "nominal", "absolute_catastrophic_reduction", fallback["nominal_benefit_interval"]["estimate"], fallback["nominal_benefit_interval"])

    for row in report["negative_result_table"]:
        append("claim_decision", row["mechanism"], row["mechanism"], "claim_decision", row["claim_decision"], row)
    for gate, passed in report["gates"].items():
        append("gate", "governance", gate, "passed", int(passed), "")
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
            errors.append("stored lane evidence does not match current native evidence")
        if report["negative_result_table"] != _negative_result_table(expected_lanes):
            errors.append("negative-result claim table drifted from lane evidence")
        recomputed = _compute_gates(report, parents)
        if report.get("gates") != recomputed:
            errors.append("stored gates do not match recomputed evidence gates")
        if not recomputed or not all(recomputed.values()):
            errors.append("one or more evidence gates failed")
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
    fields = ("row_type", "mechanism", "record_id", "metric", "value", "detail")
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
                "gates": report["gate_summary"],
                "source_rows": report["source_data"]["row_count"],
                "results": {
                    mechanism: report["lanes"][mechanism]["result"]
                    for mechanism in MECHANISMS
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CNN_ASSETS",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "MECHANISMS",
    "PARENT_ARTIFACTS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "TASK_ID",
    "build_report",
    "load_parent_artifacts",
    "source_rows",
    "validate_artifact",
    "write_report",
]
