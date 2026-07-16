"""T3.2.9 finite-horizon causal trajectory-lookup control reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from physics.trajectory_lookup_control_oracle import (
    ACTION_CONTRACT_ID,
    CONTROL_ORACLE_ROLE_ID,
    LOOKUP_SCOPE,
    CausalHistoryLookupPolicy,
    TrajectoryLookupConfig,
    build_policy,
    config_to_dict,
    enumerate_terminal_trajectories,
    evaluate_exact_policy,
    expand_open_loop_as_lookup,
    load_policy_from_state,
    optimize_policy_once,
    optimize_policy_multistart,
    resource_growth_row,
    standard_nominal_policy,
    validate_production_design,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal recovery interpreter.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t3_2_9_trajectory_lookup_control_oracle.json")
DEFAULT_CHECKPOINT = Path("docs/t3_2_9_trajectory_lookup_control_oracle.pt")
DEFAULT_SOURCE_DATA = Path("docs/t3_2_9_trajectory_lookup_control_oracle.csv")
PAPER_SOURCE = Path(
    "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
    "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
)
PAPER_FRAGMENTS = (
    "lookup table grows exponentially with time",
    "upper limit threshold for the parameters of the sBs protocol of a given model",
)
ARTIFACT_SCHEMA = "T329-CAUSAL-LOOKUP-CONTROL-ORACLE-V1"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("T3.2.9 production requires the DLEnv PyTorch environment")
    return torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        ROOT / "physics/trajectory_lookup_control_oracle.py",
        ROOT / "physics/differentiable_sbs_trajectory.py",
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    array = value.detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _trace_sha256(trace: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(trace), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_summary(run: Any, restart_index: int) -> dict[str, Any]:
    payload = asdict(run)
    trace = payload.pop("trace")
    payload["restart_index"] = restart_index
    payload["trace_sha256"] = _trace_sha256(trace)
    payload["trace_points"] = len(trace)
    payload["last_25_epoch_gain"] = float(
        trace[-1]["expected_fidelity"]
        - trace[max(0, len(trace) - 26)]["expected_fidelity"]
    )
    return payload


def _evaluation_summary(evaluation: Any) -> dict[str, Any]:
    payload = asdict(evaluation)
    payload.pop("branch_rows")
    return payload


def _refine_all_restarts(
    config: TrajectoryLookupConfig,
    family: str,
    phase_one_state: Mapping[str, Any],
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    refinement = replace(
        config,
        epochs=config.refinement_epochs,
        learning_rate=config.refinement_learning_rate,
    )
    states = tuple(phase_one_state["all_restart_states"])
    if len(states) != len(config.restart_seeds):
        raise RuntimeError("phase-one restart state count drifted")
    runs = []
    refined_states = []
    for restart_index, (seed, state) in enumerate(zip(config.restart_seeds, states)):
        if int(state["seed"]) != int(seed):
            raise RuntimeError("phase-one restart seed order drifted")
        run, refined = optimize_policy_once(
            refinement,
            family,  # type: ignore[arg-type]
            seed=seed,
            initialization_std=0.0,
            initial_raw_table=state["raw_table"],
        )
        refined = dict(refined)
        refined["phase_one_best_epoch"] = int(state["best_epoch"])
        refined["phase_one_best_expected_fidelity"] = float(
            state["best_expected_fidelity"]
        )
        refined["restart_index"] = restart_index
        runs.append(run)
        refined_states.append(refined)
    selected_index = int(
        np.argmax([run.best_expected_fidelity for run in runs])
    )
    selected = dict(refined_states[selected_index])
    selected["selected_restart_index"] = selected_index
    selected["all_restart_states"] = tuple(refined_states)
    selected["phase_one"] = phase_one_state
    return tuple(runs), selected


def _copy_lookup_policy(config: TrajectoryLookupConfig, raw_table: Any) -> Any:
    th = _require_torch()
    policy = build_policy(
        config,
        "causal_history_lookup",
        seed=0,
        initialization_std=0.0,
    )
    with th.no_grad():
        policy.raw_table.copy_(
            th.as_tensor(
                raw_table,
                dtype=policy.raw_table.dtype,
                device=policy.raw_table.device,
            )
        )
    return policy


def _source_rows(
    config: TrajectoryLookupConfig,
    phase_runs: Mapping[str, Mapping[str, Sequence[Any]]],
    evaluations: Mapping[str, Mapping[int, Any]],
    resources: Sequence[Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, by_phase in phase_runs.items():
        for phase, runs in by_phase.items():
            for restart_index, run in enumerate(runs):
                for point in run.trace:
                    rows.append(
                        {
                            "row_type": "optimization_trace",
                            "family": family,
                            "strategy": "",
                            "phase": phase,
                            "seed": run.seed,
                            "restart_index": restart_index,
                            "epoch": point["epoch"],
                            "cutoff": config.cutoff,
                            "full_cycles": config.full_cycles,
                            "half_cycles": config.half_cycles,
                            "history_nodes": run.action_node_count,
                            "terminal_branches": config.branch_count,
                            "action_scalars": run.action_node_count * 15,
                            "expected_fidelity": point["expected_fidelity"],
                            "gradient_norm": point["gradient_norm"],
                        }
                    )
    for strategy, by_cutoff in evaluations.items():
        for cutoff, evaluation in by_cutoff.items():
            for branch in evaluation.branch_rows:
                rows.append(
                    {
                        "row_type": "evaluation_branch",
                        "family": evaluation.family,
                        "strategy": strategy,
                        "cutoff": cutoff,
                        "full_cycles": config.full_cycles,
                        "half_cycles": config.half_cycles,
                        "history_nodes": (
                            config.lookup_node_count
                            if strategy == "lookup"
                            else config.half_cycles
                        ),
                        "terminal_branches": config.branch_count,
                        "action_scalars": (
                            config.lookup_node_count * 15
                            if strategy == "lookup"
                            else config.half_cycles * 15
                        ),
                        "expected_fidelity": evaluation.expected_fidelity,
                        "trajectory": branch["trajectory"],
                        "probability": branch["probability"],
                        "final_fidelity": branch["final_fidelity"],
                        "final_logical_z_signal": branch["final_logical_z_signal"],
                        "final_code_survival": branch["final_code_survival"],
                    }
                )
    for resource in resources:
        rows.append(
            {
                "row_type": "resource_growth",
                "full_cycles": resource["full_cycles"],
                "half_cycles": resource["half_cycles"],
                "history_nodes": resource["causal_history_nodes"],
                "terminal_branches": resource["terminal_branches"],
                "action_scalars": resource["lookup_action_scalars"],
                "float64_table_bytes": resource["float64_table_bytes"],
                "adam_bytes_lower_bound": resource[
                    "adam_parameter_gradient_moment_bytes_lower_bound"
                ],
                "state_bytes_lower_bound": resource[
                    "complex128_terminal_state_bytes_lower_bound"
                ],
            }
        )
    return rows


SOURCE_FIELDS = (
    "row_type",
    "family",
    "strategy",
    "phase",
    "seed",
    "restart_index",
    "epoch",
    "cutoff",
    "full_cycles",
    "half_cycles",
    "history_nodes",
    "terminal_branches",
    "action_scalars",
    "expected_fidelity",
    "gradient_norm",
    "trajectory",
    "probability",
    "final_fidelity",
    "final_logical_z_signal",
    "final_code_survival",
    "float64_table_bytes",
    "adam_bytes_lower_bound",
    "state_bytes_lower_bound",
)


def _write_source_data(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SOURCE_FIELDS})


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    th = _require_torch()
    try:
        return th.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch compatibility.
        return th.load(path, map_location="cpu")


def run_trajectory_lookup_control_oracle(
    config: TrajectoryLookupConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    th = _require_torch()
    actual = TrajectoryLookupConfig() if config is None else config
    validate_production_design(actual)
    source_text = (ROOT / PAPER_SOURCE).read_text(encoding="utf-8")
    missing_fragments = [fragment for fragment in PAPER_FRAGMENTS if fragment not in source_text]
    if missing_fragments:
        raise RuntimeError(f"paper lookup-table anchors drifted: {missing_fragments}")

    standard = standard_nominal_policy(actual)
    standard_primary = evaluate_exact_policy(actual, standard, cutoff=actual.cutoff)

    open_phase_one_runs, open_phase_one_state = optimize_policy_multistart(
        actual, "time_indexed_open_loop"
    )
    open_refinement_runs, open_state = _refine_all_restarts(
        actual, "time_indexed_open_loop", open_phase_one_state
    )
    open_policy = load_policy_from_state(actual, open_state)
    open_primary = evaluate_exact_policy(actual, open_policy, cutoff=actual.cutoff)

    expanded_open = expand_open_loop_as_lookup(actual, open_state["raw_table"])
    warm_lookup = _copy_lookup_policy(actual, expanded_open)
    warm_lookup_primary = evaluate_exact_policy(
        actual, warm_lookup, cutoff=actual.cutoff
    )
    lookup_phase_one_runs, lookup_phase_one_state = optimize_policy_multistart(
        actual,
        "causal_history_lookup",
        first_initial_raw_table=expanded_open,
    )
    lookup_refinement_runs, lookup_state = _refine_all_restarts(
        actual, "causal_history_lookup", lookup_phase_one_state
    )
    lookup_policy = load_policy_from_state(actual, lookup_state)
    lookup_primary = evaluate_exact_policy(actual, lookup_policy, cutoff=actual.cutoff)

    evaluations = {
        "standard": {
            actual.cutoff: standard_primary,
            actual.confirmation_cutoff: evaluate_exact_policy(
                actual, standard, cutoff=actual.confirmation_cutoff
            ),
        },
        "time_indexed_open_loop": {
            actual.cutoff: open_primary,
            actual.confirmation_cutoff: evaluate_exact_policy(
                actual, open_policy, cutoff=actual.confirmation_cutoff
            ),
        },
        "lookup": {
            actual.cutoff: lookup_primary,
            actual.confirmation_cutoff: evaluate_exact_policy(
                actual, lookup_policy, cutoff=actual.confirmation_cutoff
            ),
        },
    }

    checkpoint = {
        "schema": ARTIFACT_SCHEMA,
        "config": config_to_dict(actual),
        "implementation_sha256": implementation_sha256(),
        "open_loop": open_state,
        "lookup": lookup_state,
    }
    checkpoint_target = ROOT / Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    th.save(checkpoint, checkpoint_target)
    checkpoint_hash = _sha256(checkpoint_target)
    loaded = _load_checkpoint(checkpoint_target)
    replay_open = load_policy_from_state(actual, loaded["open_loop"])
    replay_lookup = load_policy_from_state(actual, loaded["lookup"])
    replay_values = {
        "open_primary": evaluate_exact_policy(
            actual, replay_open, cutoff=actual.cutoff
        ).expected_fidelity,
        "lookup_primary": evaluate_exact_policy(
            actual, replay_lookup, cutoff=actual.cutoff
        ).expected_fidelity,
        "open_confirmation": evaluate_exact_policy(
            actual, replay_open, cutoff=actual.confirmation_cutoff
        ).expected_fidelity,
        "lookup_confirmation": evaluate_exact_policy(
            actual, replay_lookup, cutoff=actual.confirmation_cutoff
        ).expected_fidelity,
    }

    resources = [resource_growth_row(cycle, cutoff=actual.cutoff) for cycle in range(1, 11)]
    phase_runs = {
        "time_indexed_open_loop": {
            "phase_one": open_phase_one_runs,
            "refinement": open_refinement_runs,
        },
        "causal_history_lookup": {
            "phase_one": lookup_phase_one_runs,
            "refinement": lookup_refinement_runs,
        },
    }
    rows = _source_rows(actual, phase_runs, evaluations, resources)
    source_target = ROOT / Path(source_data_path)
    _write_source_data(source_target, rows)

    outcomes = enumerate_terminal_trajectories(actual.half_cycles, device=actual.device)
    lookup_index_audit = CausalHistoryLookupPolicy(
        actual.half_cycles,
        device=actual.device,
        dtype=th.float64,
    )
    prefix_coverage = []
    for depth in range(actual.half_cycles):
        unique = sorted(
            set(
                lookup_index_audit.node_indices(outcomes[:, :depth], depth)
                .detach()
                .cpu()
                .tolist()
            )
        )
        prefix_coverage.append(
            {
                "depth": depth,
                "unique_indices": unique,
                "expected_start": 2**depth - 1,
                "expected_stop_exclusive": 2 ** (depth + 1) - 1,
            }
        )

    probability_evaluations = [
        evaluation
        for by_cutoff in evaluations.values()
        for evaluation in by_cutoff.values()
    ]
    phase_one_summaries = {
        "time_indexed_open_loop": [
            _run_summary(run, index)
            for index, run in enumerate(open_phase_one_runs)
        ],
        "causal_history_lookup": [
            _run_summary(run, index)
            for index, run in enumerate(lookup_phase_one_runs)
        ],
    }
    refinement_summaries = {
        "time_indexed_open_loop": [
            _run_summary(run, index)
            for index, run in enumerate(open_refinement_runs)
        ],
        "causal_history_lookup": [
            _run_summary(run, index)
            for index, run in enumerate(lookup_refinement_runs)
        ],
    }
    expected_source_rows = (
        2
        * len(actual.restart_seeds)
        * (actual.epochs + actual.refinement_epochs + 2)
        + 3 * 2 * actual.branch_count
        + 10
    )
    gates = {
        "paper_lookup_and_exponential_anchors_are_live": not missing_fragments,
        "role_is_control_oracle_not_decoder_oracle": actual.role_id
        == CONTROL_ORACLE_ROLE_ID,
        "same_bounded_fifteen_action_contract_is_used": actual.action_contract_id
        == ACTION_CONTRACT_ID,
        "two_cycle_tree_has_fifteen_prefix_nodes_and_sixteen_branches": actual.lookup_node_count
        == 15
        and actual.branch_count == 16,
        "every_causal_prefix_node_is_covered_exactly_by_depth": all(
            item["unique_indices"]
            == list(range(item["expected_start"], item["expected_stop_exclusive"]))
            for item in prefix_coverage
        ),
        "all_exact_trajectory_distributions_normalize": all(
            abs(item.trajectory_probability_sum - 1.0) <= 2.0e-10
            for item in probability_evaluations
        ),
        "all_terminal_branches_have_positive_probability": all(
            item.minimum_trajectory_probability > 0.0
            for item in probability_evaluations
        ),
        "density_diagnostics_pass": all(
            item.maximum_trace_error <= 2.0e-10
            and item.maximum_hermiticity_error <= 2.0e-10
            and item.minimum_final_eigenvalue >= -2.0e-9
            for item in probability_evaluations
        ),
        "optimized_open_loop_improves_nominal_initialization": open_primary.expected_fidelity
        >= standard_primary.expected_fidelity + 1.0e-3,
        "open_loop_embeds_exactly_in_lookup_tree": abs(
            warm_lookup_primary.expected_fidelity - open_primary.expected_fidelity
        )
        <= 2.0e-10,
        "optimized_lookup_retains_its_nested_open_loop_reference": lookup_primary.expected_fidelity
        >= open_primary.expected_fidelity - 2.0e-10,
        "every_restart_uses_full_node_gradient_coverage": all(
            summary["gradient_covered_nodes"] == summary["action_node_count"]
            for summaries in (
                list(phase_one_summaries.values())
                + list(refinement_summaries.values())
            )
            for summary in summaries
        ),
        "every_restart_changes_every_action_node": all(
            summary["changed_nodes"] == summary["action_node_count"]
            for summaries in (
                list(phase_one_summaries.values())
                + list(refinement_summaries.values())
            )
            for summary in summaries
        ),
        "three_or_more_restarts_are_retained_for_each_family": all(
            len(summaries) >= 3 for summaries in refinement_summaries.values()
        ),
        "selected_refinements_have_small_tail_gain": all(
            refinement_summaries[family][selected]["last_25_epoch_gain"]
            <= 2.0e-4
            for family, selected in (
                ("time_indexed_open_loop", int(open_state["selected_restart_index"])),
                ("causal_history_lookup", int(lookup_state["selected_restart_index"])),
            )
        ),
        "confirmation_cutoff_is_evaluated_after_selection_without_retuning": actual.confirmation_cutoff
        != actual.cutoff
        and all(actual.confirmation_cutoff in lane for lane in evaluations.values()),
        "checkpoint_reload_reproduces_all_selected_fidelities": max(
            abs(replay_values["open_primary"] - open_primary.expected_fidelity),
            abs(replay_values["lookup_primary"] - lookup_primary.expected_fidelity),
            abs(
                replay_values["open_confirmation"]
                - evaluations["time_indexed_open_loop"][actual.confirmation_cutoff].expected_fidelity
            ),
            abs(
                replay_values["lookup_confirmation"]
                - evaluations["lookup"][actual.confirmation_cutoff].expected_fidelity
            ),
        )
        <= 2.0e-10,
        "resource_projection_records_exact_exponential_growth": resources[-1][
            "terminal_branches"
        ]
        == 2**20
        and resources[-1]["causal_history_nodes"] == 2**20 - 1,
        "source_data_contains_full_traces_branches_and_resources": len(rows)
        == expected_source_rows,
        "no_global_optimum_or_deployment_claim_is_made": "globally certified optimum"
        in LOOKUP_SCOPE
        and "target-board" in LOOKUP_SCOPE,
    }
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "task_id": "T3.2.9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "config": config_to_dict(actual),
        "literature": {
            "source": PAPER_SOURCE.as_posix(),
            "source_sha256": _sha256(ROOT / PAPER_SOURCE),
            "required_fragments": list(PAPER_FRAGMENTS),
            "scope_note": (
                "paper lookup table motivates the finite-horizon ansatz reference; "
                "the present numbers use this repository's model and are not Fig.S4 reproduction"
            ),
        },
        "tree_contract": {
            "outcome_alphabet": {"g": 0, "e": 1},
            "decision_timing": "action at half-cycle j sees exactly outcomes [0,j)",
            "terminal_branches": actual.branch_count,
            "causal_history_nodes": actual.lookup_node_count,
            "lookup_action_scalars": actual.lookup_node_count * 15,
            "prefix_coverage": prefix_coverage,
        },
        "optimization": {
            "objective": "exact sum_over_trajectories P_theta(m) * final_state_fidelity",
            "selection_rule": "highest exact optimization-model fidelity across all retained restarts",
            "evaluation_not_used_for_selection": True,
            "phase_one": phase_one_summaries,
            "refinement": refinement_summaries,
            "selected_open_loop_restart": int(open_state["selected_restart_index"]),
            "selected_lookup_restart": int(lookup_state["selected_restart_index"]),
            "selected_open_loop_table_sha256": _tensor_sha256(open_state["raw_table"]),
            "selected_lookup_table_sha256": _tensor_sha256(lookup_state["raw_table"]),
            "warm_start_lookup_fidelity": warm_lookup_primary.expected_fidelity,
        },
        "evaluations": {
            strategy: {
                str(cutoff): _evaluation_summary(evaluation)
                for cutoff, evaluation in by_cutoff.items()
            }
            for strategy, by_cutoff in evaluations.items()
        },
        "comparisons": {
            "primary_lookup_minus_standard_fidelity": lookup_primary.expected_fidelity
            - standard_primary.expected_fidelity,
            "primary_lookup_minus_optimized_open_loop_fidelity": lookup_primary.expected_fidelity
            - open_primary.expected_fidelity,
            "primary_lookup_minus_standard_logical_z": lookup_primary.expected_logical_z_signal
            - standard_primary.expected_logical_z_signal,
            "primary_lookup_minus_standard_ground_fraction": lookup_primary.expected_ground_outcome_fraction
            - standard_primary.expected_ground_outcome_fraction,
            "confirmation_lookup_minus_standard_fidelity": evaluations["lookup"][
                actual.confirmation_cutoff
            ].expected_fidelity
            - evaluations["standard"][actual.confirmation_cutoff].expected_fidelity,
        },
        "resource_growth": resources,
        "measured_optimization_wall_time_seconds": {
            family: {
                phase: float(sum(run.wall_time_seconds for run in runs))
                for phase, runs in by_phase.items()
            }
            for family, by_phase in phase_runs.items()
        },
        "checkpoint": {
            "path": Path(checkpoint_path).as_posix(),
            "sha256": checkpoint_hash,
            "schema": ARTIFACT_SCHEMA,
            "replay_values": replay_values,
        },
        "source_data": {
            "path": Path(source_data_path).as_posix(),
            "sha256": _sha256(source_target),
            "row_count": len(rows),
            "row_types": sorted({row["row_type"] for row in rows}),
        },
        "implementation_sha256": implementation_sha256(),
        "gates": gates,
        "required_gates": list(gates),
        "claim_boundary": {
            "allowed": (
                "finite-horizon finite-cutoff assumed-model causal lookup control-policy reference "
                "with exact branch enumeration and empirical multi-start optimization"
            ),
            "forbidden": (
                "globally certified ansatz optimum",
                "decoder oracle or channel-recovery bound",
                "long-horizon deployable lookup controller",
                "paper Fig.S4 numerical reproduction",
                "multilevel pulse device or target-board result",
            ),
        },
    }
    artifact_target = ROOT / Path(artifact_path)
    artifact_target.parent.mkdir(parents=True, exist_ok=True)
    artifact_target.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if artifact["status"] != "PASS":
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"T3.2.9 failed gates: {failed}")
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run_trajectory_lookup_control_oracle(
        TrajectoryLookupConfig(epochs=args.epochs, device=args.device),
        artifact_path=args.artifact,
        checkpoint_path=args.checkpoint,
        source_data_path=args.source_data,
    )
    print(json.dumps({
        "status": artifact["status"],
        "gates": len(artifact["gates"]),
        "lookup_minus_standard": artifact["comparisons"]["primary_lookup_minus_standard_fidelity"],
        "lookup_minus_open_loop": artifact["comparisons"]["primary_lookup_minus_optimized_open_loop_fidelity"],
        "source_rows": artifact["source_data"]["row_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SOURCE_DATA",
    "PAPER_SOURCE",
    "ARTIFACT_SCHEMA",
    "implementation_sha256",
    "run_trajectory_lookup_control_oracle",
    "main",
]
