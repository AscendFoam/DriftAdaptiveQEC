"""T3.2.10 PRL-inspired exponential-recurrence baseline validation.

The artifact keeps two non-interchangeable evidence lanes:

* exact finite-cutoff two-cycle physical fidelity for a learned 15-vector
  g/e recurrence, compared with standard control and the T3.2.9 lookup oracle;
* abstract synthetic event-control cost for the same scalar recurrence kernel,
  compared with the T3.2.5 run-length FSM on identical frozen traces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.continuous_adaptive_map import _mean_interval
from cnn_fpga.benchmark.run_length_fsm_baseline import (
    CONTROLLERS,
    RunLengthValidationConfig,
    _controller_metrics,
    _evaluate_trace,
    _fit_thresholds,
    _make_trace,
    _seed_cluster_difference,
    event_scenarios,
)
from cnn_fpga.runtime.exponential_recurrence import (
    ExponentialEventControllerConfig,
    ExponentialRecurrenceEventController,
)
from cnn_fpga.runtime.run_length_fsm import RunLengthFSMInput
from physics.exponential_recurrence_control import (
    ExponentialRecurrenceConfig,
    FixedPointExponentialPolicy,
    config_to_dict,
    load_policy_state,
    optimize_recurrence_multistart,
    state_dict_sha256,
    validate_production_design,
)
from physics.trajectory_lookup_control_oracle import (
    TrajectoryLookupConfig,
    evaluate_exact_policy,
    load_policy_from_state,
    standard_nominal_policy,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t3_2_10_exponential_recurrence_validation.json")
DEFAULT_CHECKPOINT = Path("docs/t3_2_10_exponential_recurrence_checkpoints.pt")
DEFAULT_SOURCE_DATA = Path("docs/t3_2_10_exponential_recurrence_source_data.csv")
LOOKUP_CHECKPOINT = Path("docs/t3_2_9_trajectory_lookup_control_oracle.pt")
PAPER_SOURCE = Path(
    "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
    "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
)
PAPER_FRAGMENTS = (
    "exponential saturation",
    "parameters of the sBs protocol",
    "decay rate",
)
SCHEMA = "T3210-EXPONENTIAL-RECURRENCE-BASELINE-V1"
EVENT_CONTROLLERS = CONTROLLERS + ("exponential_recurrence", "exponential_recurrence_fixed_point")


def _require_torch() -> Any:
    import torch

    return torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "physics/exponential_recurrence_control.py",
        "physics/trajectory_lookup_control_oracle.py",
        "cnn_fpga/runtime/exponential_recurrence.py",
        "cnn_fpga/runtime/run_length_fsm.py",
        "cnn_fpga/benchmark/run_length_fsm_baseline.py",
        "cnn_fpga/benchmark/exponential_recurrence_baseline.py",
    ):
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EventRecurrenceValidationConfig:
    training_seeds: tuple[int, ...] = (20261101, 20261102, 20261103)
    evaluation_seeds: tuple[int, ...] = tuple(range(20261121, 20261129))
    training_cycles: int = 4096
    evaluation_cycles: int = 12_000
    decay_g_grid: tuple[float, ...] = (0.45, 0.65, 0.80)
    decay_e_grid: tuple[float, ...] = (0.30, 0.55, 0.75)
    decay_leakage_grid: tuple[float, ...] = (0.15, 0.40)
    recovery_enter_grid: tuple[float, ...] = (0.45, 0.65)
    leakage_enter_grid: tuple[float, ...] = (0.35, 0.60)
    recovery_exit: float = 0.25
    leakage_exit: float = 0.15
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        reference = RunLengthValidationConfig(
            training_seeds=self.training_seeds,
            evaluation_seeds=self.evaluation_seeds,
            training_cycles=self.training_cycles,
            evaluation_cycles=self.evaluation_cycles,
            confidence_level=self.confidence_level,
        )
        object.__setattr__(self, "training_seeds", reference.training_seeds)
        object.__setattr__(self, "evaluation_seeds", reference.evaluation_seeds)
        object.__setattr__(self, "training_cycles", reference.training_cycles)
        object.__setattr__(self, "evaluation_cycles", reference.evaluation_cycles)
        object.__setattr__(self, "confidence_level", reference.confidence_level)
        for name in (
            "decay_g_grid",
            "decay_e_grid",
            "decay_leakage_grid",
            "recovery_enter_grid",
            "leakage_enter_grid",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or len(set(values)) != len(values) or any(not 0.0 < value < 1.0 for value in values):
                raise ValueError(f"{name} must contain unique values in (0,1)")
            object.__setattr__(self, name, values)
        if not 0.0 <= self.recovery_exit < min(self.recovery_enter_grid):
            raise ValueError("recovery_exit must be below every recovery-enter candidate")
        if not 0.0 <= self.leakage_exit < min(self.leakage_enter_grid):
            raise ValueError("leakage_exit must be below every leakage-enter candidate")

    @property
    def grid_size(self) -> int:
        return int(
            np.prod(
                [
                    len(self.decay_g_grid),
                    len(self.decay_e_grid),
                    len(self.decay_leakage_grid),
                    len(self.recovery_enter_grid),
                    len(self.leakage_enter_grid),
                ]
            )
        )


def _event_candidates(settings: EventRecurrenceValidationConfig) -> tuple[ExponentialEventControllerConfig, ...]:
    return tuple(
        ExponentialEventControllerConfig(
            decay_g=decay_g,
            decay_e=decay_e,
            decay_leakage=decay_leakage,
            recovery_enter=recovery_enter,
            recovery_exit=settings.recovery_exit,
            leakage_enter=leakage_enter,
            leakage_exit=settings.leakage_exit,
        )
        for decay_g, decay_e, decay_leakage, recovery_enter, leakage_enter in itertools.product(
            settings.decay_g_grid,
            settings.decay_e_grid,
            settings.decay_leakage_grid,
            settings.recovery_enter_grid,
            settings.leakage_enter_grid,
        )
    )


def _fsm_input(trace: Any, cycle: int) -> RunLengthFSMInput:
    observed = trace.stream.steps[cycle].observed
    valid, crc_ok, parameter_fresh, deadline_ok = trace.health[cycle]
    return RunLengthFSMInput(
        cycle_index=cycle,
        residual=observed.residual_syndrome,
        syndrome_x=observed.syndrome.x,
        syndrome_z=observed.syndrome.z,
        quadrature_phase_bit=cycle & 1,
        valid=valid,
        crc_ok=crc_ok,
        parameter_fresh=parameter_fresh,
        deadline_ok=deadline_ok,
    )


def _replay_recurrence(
    trace: Any,
    config: ExponentialEventControllerConfig,
    *,
    arithmetic: str,
) -> tuple[tuple[str, ...], int, float, int]:
    controller = ExponentialRecurrenceEventController(config, arithmetic=arithmetic)  # type: ignore[arg-type]
    modes: list[str] = []
    maximum_state_error = 0.0
    conflicts = 0
    floating_reference = (
        ExponentialRecurrenceEventController(config, arithmetic="float64")
        if arithmetic == "fixed_point"
        else None
    )
    for cycle in range(len(trace.stream.steps)):
        event = _fsm_input(trace, cycle)
        decision = controller.step(event)
        modes.append(decision.mode)
        conflicts += int(decision.bank_conflict)
        if floating_reference is not None:
            reference = floating_reference.step(event)
            maximum_state_error = max(
                maximum_state_error,
                abs(decision.x_state - reference.x_state),
                abs(decision.z_state - reference.z_state),
                abs(decision.leakage_state - reference.leakage_state),
            )
    return tuple(modes), controller.bank_writes, maximum_state_error, conflicts


def _fit_event_recurrence(
    traces: Sequence[Any], settings: EventRecurrenceValidationConfig
) -> tuple[ExponentialEventControllerConfig, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for candidate in _event_candidates(settings):
        costs = []
        writes = []
        for trace in traces:
            modes, bank_writes, _, conflicts = _replay_recurrence(trace, candidate, arithmetic="float64")
            if conflicts:
                raise RuntimeError("training recurrence unexpectedly hit a parameter-bank conflict")
            metrics = _controller_metrics(trace.truth_modes, modes, bank_writes=bank_writes)
            costs.append(float(metrics["event_plus_write_cost"]))
            writes.append(bank_writes)
        rows.append(
            {
                "decay_g": candidate.decay_g,
                "decay_e": candidate.decay_e,
                "decay_leakage": candidate.decay_leakage,
                "recovery_enter": candidate.recovery_enter,
                "recovery_exit": candidate.recovery_exit,
                "leakage_enter": candidate.leakage_enter,
                "leakage_exit": candidate.leakage_exit,
                "training_mean_event_plus_write_cost": float(np.mean(costs)),
                "training_mean_bank_writes": float(np.mean(writes)),
                "training_traces": len(traces),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["training_mean_event_plus_write_cost"],
            row["training_mean_bank_writes"],
            row["decay_g"],
            row["decay_e"],
            row["decay_leakage"],
            row["recovery_enter"],
            row["leakage_enter"],
        ),
    )
    config = ExponentialEventControllerConfig(
        decay_g=float(selected["decay_g"]),
        decay_e=float(selected["decay_e"]),
        decay_leakage=float(selected["decay_leakage"]),
        recovery_enter=float(selected["recovery_enter"]),
        recovery_exit=float(selected["recovery_exit"]),
        leakage_enter=float(selected["leakage_enter"]),
        leakage_exit=float(selected["leakage_exit"]),
    )
    return config, rows


def _event_lane(settings: EventRecurrenceValidationConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenarios = event_scenarios()
    training = [
        _make_trace(scenario, seed, settings.training_cycles, scenario_index)
        for scenario_index, scenario in enumerate(scenarios)
        for seed in settings.training_seeds
    ]
    selected_recurrence, recurrence_grid = _fit_event_recurrence(training, settings)
    fsm_settings = RunLengthValidationConfig(
        training_seeds=settings.training_seeds,
        evaluation_seeds=settings.evaluation_seeds,
        training_cycles=settings.training_cycles,
        evaluation_cycles=settings.evaluation_cycles,
        confidence_level=settings.confidence_level,
    )
    selected_fsm, fsm_grid = _fit_thresholds(training, fsm_settings)
    rows: list[dict[str, Any]] = []
    mode_mismatches = 0
    maximum_state_error = 0.0
    for scenario_index, scenario in enumerate(scenarios):
        for seed in settings.evaluation_seeds:
            trace = _make_trace(scenario, seed, settings.evaluation_cycles, scenario_index)
            row, _ = _evaluate_trace(trace, selected_fsm)
            float_modes, float_writes, _, float_conflicts = _replay_recurrence(
                trace, selected_recurrence, arithmetic="float64"
            )
            fixed_modes, fixed_writes, state_error, fixed_conflicts = _replay_recurrence(
                trace, selected_recurrence, arithmetic="fixed_point"
            )
            mode_mismatches += sum(a != b for a, b in zip(float_modes, fixed_modes, strict=True))
            maximum_state_error = max(maximum_state_error, state_error)
            for name, modes, writes in (
                ("exponential_recurrence", float_modes, float_writes),
                ("exponential_recurrence_fixed_point", fixed_modes, fixed_writes),
            ):
                for metric, value in _controller_metrics(trace.truth_modes, modes, bank_writes=writes).items():
                    row[f"{name}_{metric}"] = value
            row["recurrence_float_bank_conflicts"] = float_conflicts
            row["recurrence_fixed_bank_conflicts"] = fixed_conflicts
            row["recurrence_fixed_mode_mismatches"] = sum(
                a != b for a, b in zip(float_modes, fixed_modes, strict=True)
            )
            row["recurrence_fixed_maximum_state_error"] = state_error
            rows.append(row)
    metrics = (
        "event_cost",
        "event_plus_write_cost",
        "action_accuracy",
        "unsafe_miss_rate",
        "false_intervention_rate",
        "mean_event_detection_delay_cycles",
        "bank_writes",
    )
    aggregate = {
        controller: {
            metric: float(np.mean([float(row[f"{controller}_{metric}"]) for row in rows]))
            for metric in metrics
        }
        for controller in EVENT_CONTROLLERS
    }
    comparisons = {
        "run_length_minus_recurrence": _mean_interval(
            _seed_cluster_difference(rows, "run_length_fsm", "exponential_recurrence"),
            settings.confidence_level,
        ),
        "memoryless_minus_recurrence": _mean_interval(
            _seed_cluster_difference(rows, "memoryless_event", "exponential_recurrence"),
            settings.confidence_level,
        ),
        "recurrence_minus_truth": _mean_interval(
            _seed_cluster_difference(rows, "exponential_recurrence", "truth_oracle"),
            settings.confidence_level,
        ),
    }
    cycles = sum(int(row["cycles"]) for row in rows)
    return (
        {
            "metric_domain": "abstract_event_control_cost_not_physical_fidelity_or_LER",
            "validation_config": asdict(settings),
            "training": {
                "evaluation_truth_used": False,
                "training_traces": len(training),
                "training_cycles": len(training) * settings.training_cycles,
                "recurrence_grid_size": len(recurrence_grid),
                "recurrence_replay_cycles": len(training) * settings.training_cycles * len(recurrence_grid),
                "selected_recurrence": asdict(selected_recurrence),
                "recurrence_grid": recurrence_grid,
                "selected_run_length_fsm": asdict(selected_fsm),
                "run_length_grid": fsm_grid,
            },
            "evaluation": {
                "traces": len(rows),
                "cycles": cycles,
                "aggregate": aggregate,
                "paired_seed_cluster_comparisons": comparisons,
                "fixed_point": {
                    "state_format": "Q4.16 signed state; Q2.18 decay coefficient",
                    "mode_mismatches": mode_mismatches,
                    "mode_parity": 1.0 - mode_mismatches / cycles,
                    "maximum_state_error": maximum_state_error,
                    "bank_conflicts": sum(
                        int(row["recurrence_float_bank_conflicts"]) + int(row["recurrence_fixed_bank_conflicts"])
                        for row in rows
                    ),
                },
            },
        },
        rows,
    )


def _lookup_config(config: ExponentialRecurrenceConfig) -> TrajectoryLookupConfig:
    return TrajectoryLookupConfig(
        full_cycles=config.full_cycles,
        cutoff=config.cutoff,
        confirmation_cutoff=config.confirmation_cutoff,
        projector_delta=config.projector_delta,
        cavity_lifetime_us=config.cavity_lifetime_us,
        ancilla_t1_us=config.ancilla_t1_us,
        ancilla_t2_us=config.ancilla_t2_us,
        device=config.device,
        real_dtype=config.real_dtype,
    )


def _load_torch(path: Path) -> Mapping[str, Any]:
    torch = _require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _evaluation_summary(value: Any) -> dict[str, Any]:
    return {name: getattr(value, name) for name in (
        "family",
        "cutoff",
        "expected_fidelity",
        "expected_logical_z_signal",
        "expected_code_survival",
        "expected_ground_outcome_fraction",
        "trajectory_probability_sum",
        "minimum_trajectory_probability",
        "maximum_trajectory_probability",
        "maximum_trace_error",
        "maximum_hermiticity_error",
        "minimum_final_eigenvalue",
    )}


def _optimization_summary(run: Any, restart_index: int) -> dict[str, Any]:
    payload = asdict(run)
    payload["restart_index"] = restart_index
    payload["trace"] = list(run.trace)
    return payload


def _source_rows(
    config: ExponentialRecurrenceConfig,
    phase_runs: Sequence[Any],
    refinement_runs: Sequence[Any],
    evaluations: Mapping[str, Mapping[int, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    event_lane: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, runs in (("phase_one", phase_runs), ("refinement", refinement_runs)):
        for restart, run in enumerate(runs):
            for point in run.trace:
                rows.append({
                    "row_type": "physical_optimization",
                    "metric_domain": "exact_physical_fidelity",
                    "phase": phase,
                    "restart_index": restart,
                    "seed": run.seed,
                    **point,
                })
    for strategy, by_cutoff in evaluations.items():
        for cutoff, evaluation in by_cutoff.items():
            for branch in evaluation.branch_rows:
                rows.append({
                    "row_type": "physical_terminal_branch",
                    "metric_domain": "exact_physical_fidelity",
                    "strategy": strategy,
                    "cutoff": cutoff,
                    **branch,
                })
    for index, row in enumerate(event_lane["training"]["recurrence_grid"]):
        rows.append({"row_type": "event_training_grid", "metric_domain": "abstract_event_control_cost", "grid_index": index, **row})
    for row in event_rows:
        rows.append({"row_type": "event_evaluation_trace", "metric_domain": "abstract_event_control_cost", **row})
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields_union = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_union)
        writer.writeheader()
        writer.writerows(rows)


def run_exponential_recurrence_baseline(
    config: ExponentialRecurrenceConfig | None = None,
    event_config: EventRecurrenceValidationConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    torch = _require_torch()
    actual = ExponentialRecurrenceConfig() if config is None else config
    events = EventRecurrenceValidationConfig() if event_config is None else event_config
    validate_production_design(actual)
    paper_path = ROOT / PAPER_SOURCE
    paper_text = paper_path.read_text(encoding="utf-8")
    missing = [fragment for fragment in PAPER_FRAGMENTS if fragment.lower() not in paper_text.lower()]
    if missing:
        raise RuntimeError(f"paper exponential-recurrence anchors drifted: {missing}")
    lookup_path = ROOT / LOOKUP_CHECKPOINT
    if not lookup_path.exists():
        raise FileNotFoundError("T3.2.9 lookup checkpoint is required for the frozen comparator")
    lookup_checkpoint = _load_torch(lookup_path)

    phase_runs, refinement_runs, selected = optimize_recurrence_multistart(actual)
    selected_policy = load_policy_state(actual, selected["state_dict"])
    fixed_policy = FixedPointExponentialPolicy(selected_policy)
    lookup_config = _lookup_config(actual)
    standard = standard_nominal_policy(lookup_config)
    lookup_policy = load_policy_from_state(lookup_config, lookup_checkpoint["lookup"])
    evaluations: dict[str, dict[int, Any]] = {
        "standard": {},
        "exponential_recurrence": {},
        "exponential_recurrence_fixed_point": {},
        "trajectory_lookup_control_oracle": {},
    }
    for cutoff in (actual.cutoff, actual.confirmation_cutoff):
        evaluations["standard"][cutoff] = evaluate_exact_policy(lookup_config, standard, cutoff=cutoff)
        evaluations["exponential_recurrence"][cutoff] = evaluate_exact_policy(lookup_config, selected_policy, cutoff=cutoff)
        evaluations["exponential_recurrence_fixed_point"][cutoff] = evaluate_exact_policy(lookup_config, fixed_policy, cutoff=cutoff)
        evaluations["trajectory_lookup_control_oracle"][cutoff] = evaluate_exact_policy(lookup_config, lookup_policy, cutoff=cutoff)

    event_payload, event_rows = _event_lane(events)
    checkpoint_target = ROOT / Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": SCHEMA,
        "implementation_sha256": implementation_sha256(),
        "config": config_to_dict(actual),
        "selected_restart_index": selected["selected_restart_index"],
        "selected_seed": selected["seed"],
        "selected_state_dict_sha256": selected["state_dict_sha256"],
        "selected_state_dict": selected["state_dict"],
        "all_phase_one_states": selected["all_phase_one_states"],
        "all_refinement_states": selected["all_refinement_states"],
        "selected_event_config": event_payload["training"]["selected_recurrence"],
    }
    torch.save(checkpoint, checkpoint_target)
    reloaded = _load_torch(checkpoint_target)
    replay_policy = load_policy_state(actual, reloaded["selected_state_dict"])
    replay = evaluate_exact_policy(lookup_config, replay_policy, cutoff=actual.cutoff)

    source_rows = _source_rows(actual, phase_runs, refinement_runs, evaluations, event_rows, event_payload)
    source_target = ROOT / Path(source_data_path)
    _write_csv(source_target, source_rows)
    primary = actual.cutoff
    confirmation = actual.confirmation_cutoff
    recurrence_primary = evaluations["exponential_recurrence"][primary]
    standard_primary = evaluations["standard"][primary]
    lookup_primary = evaluations["trajectory_lookup_control_oracle"][primary]
    fixed_primary = evaluations["exponential_recurrence_fixed_point"][primary]
    selected_index = int(selected["selected_restart_index"])
    selected_refinement = refinement_runs[selected_index]
    trace_values = [float(row["expected_fidelity"]) for row in selected_refinement.trace]
    tail_span = min(25, len(trace_values) - 1)
    tail_gain = max(trace_values[-tail_span:]) - trace_values[-tail_span]
    max_abs_state = max(
        float(torch.max(torch.abs(value)).detach().cpu())
        for name, value in selected_policy.state_dict().items()
        if "leakage" not in name
    )
    decay = selected_policy.ge_decay().detach().cpu().numpy()
    comparisons = {
        "primary_recurrence_minus_standard_fidelity": recurrence_primary.expected_fidelity - standard_primary.expected_fidelity,
        "primary_lookup_minus_recurrence_fidelity": lookup_primary.expected_fidelity - recurrence_primary.expected_fidelity,
        "primary_fixed_minus_float_fidelity": fixed_primary.expected_fidelity - recurrence_primary.expected_fidelity,
        "confirmation_recurrence_minus_standard_fidelity": evaluations["exponential_recurrence"][confirmation].expected_fidelity - evaluations["standard"][confirmation].expected_fidelity,
        "confirmation_lookup_minus_recurrence_fidelity": evaluations["trajectory_lookup_control_oracle"][confirmation].expected_fidelity - evaluations["exponential_recurrence"][confirmation].expected_fidelity,
        "event_run_length_minus_recurrence_cost": event_payload["evaluation"]["aggregate"]["run_length_fsm"]["event_plus_write_cost"] - event_payload["evaluation"]["aggregate"]["exponential_recurrence"]["event_plus_write_cost"],
    }
    gates = {
        "paper_exponential_saturation_anchors_are_live": not missing,
        "physical_policy_has_75_trainable_and_105_stored_scalars": selected_policy.parameter_count == 75 and selected_policy.stored_scalar_count == 105,
        "all_restarts_cover_and_change_all_trainable_scalars": all(run.gradient_covered_scalars == 75 and run.changed_scalars == 75 for run in (*phase_runs, *refinement_runs)),
        "selected_refinement_tail_is_flat": tail_gain < 2.0e-4,
        "learned_decays_are_strictly_stable": bool(np.all((decay > actual.decay_minimum) & (decay < actual.decay_maximum))),
        "learned_raw_state_is_not_numerically_explosive": max_abs_state < 8.0,
        "primary_recurrence_improves_standard": comparisons["primary_recurrence_minus_standard_fidelity"] > 1.0e-3,
        "lookup_remains_upper_ansatz_reference_at_primary_cutoff": comparisons["primary_lookup_minus_recurrence_fidelity"] >= -2.0e-10,
        "fixed_point_physical_fidelity_loss_is_bounded": abs(comparisons["primary_fixed_minus_float_fidelity"]) < 1.0e-3,
        "confirmation_cutoff_is_frozen_not_retrained": evaluations["exponential_recurrence"][confirmation].cutoff == confirmation,
        "exact_branch_probabilities_normalize": all(abs(value.trajectory_probability_sum - 1.0) < 2.0e-10 for by_cutoff in evaluations.values() for value in by_cutoff.values()),
        "exact_density_diagnostics_are_physical": all(value.maximum_trace_error < 2.0e-10 and value.maximum_hermiticity_error < 2.0e-10 and value.minimum_final_eigenvalue > -2.0e-9 for by_cutoff in evaluations.values() for value in by_cutoff.values()),
        "checkpoint_replay_is_exact": abs(replay.expected_fidelity - recurrence_primary.expected_fidelity) < 2.0e-12 and reloaded["selected_state_dict_sha256"] == state_dict_sha256(reloaded["selected_state_dict"]),
        "event_grid_is_complete_training_only": event_payload["training"]["recurrence_grid_size"] == events.grid_size and not event_payload["training"]["evaluation_truth_used"],
        "event_evaluation_uses_full_frozen_workload": event_payload["evaluation"]["traces"] == 32 and event_payload["evaluation"]["cycles"] == 384_000,
        "event_fixed_point_mode_parity_is_high": event_payload["evaluation"]["fixed_point"]["mode_parity"] > 0.999,
        "event_runtime_has_no_parameter_bank_conflicts": event_payload["evaluation"]["fixed_point"]["bank_conflicts"] == 0,
        "event_and_physical_metric_domains_remain_separate": event_payload["metric_domain"] == "abstract_event_control_cost_not_physical_fidelity_or_LER",
        "source_data_is_nontrivial": len(source_rows) > 1_500,
    }
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": "T3.2.10",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": implementation_sha256(),
        "config": config_to_dict(actual),
        "literature": {
            "source": PAPER_SOURCE.as_posix(),
            "source_sha256": _sha256(paper_path),
            "required_fragments": list(PAPER_FRAGMENTS),
            "interpretation": "paper-inspired exponential saturation; repository-local optimization, not a paper-number reproduction",
        },
        "recurrence_contract": {
            "formula": "pi[t+1] = a[m] * pi[t] + (1-a[m]) * pi_inf[m]",
            "decision_timing": "action at half-cycle j sees exactly outcomes [0,j)",
            "outcomes": ["g", "e", "leakage"],
            "physical_training": "g/e only; leakage branch fixed and explicitly uncalibrated",
            "online_state_scalars": 15,
            "trainable_scalars": 75,
            "stored_scalars_including_leakage": 105,
            "horizon_scaling": "constant parameter storage; O(15) multiply-accumulates per observation",
        },
        "optimization": {
            "phase_one": [_optimization_summary(run, index) for index, run in enumerate(phase_runs)],
            "refinement": [_optimization_summary(run, index) for index, run in enumerate(refinement_runs)],
            "selected_restart_index": selected_index,
            "selected_seed": int(selected["seed"]),
            "selected_state_dict_sha256": selected["state_dict_sha256"],
            "selected_refinement_tail_gain_last_25": tail_gain,
            "learned_g_e_decay_minimum": float(np.min(decay)),
            "learned_g_e_decay_maximum": float(np.max(decay)),
            "maximum_absolute_raw_stored_value": max_abs_state,
        },
        "physical_fidelity_lane": {
            "metric_domain": "exact_finite_cutoff_two_level_sBs_physical_fidelity",
            "evaluations": {
                strategy: {str(cutoff): _evaluation_summary(value) for cutoff, value in by_cutoff.items()}
                for strategy, by_cutoff in evaluations.items()
            },
            "comparisons": comparisons,
            "fixed_point": {
                "state_format": "signed Q4.14 raw state",
                "decay_format": "unsigned Q0.16",
                "note": "software integer recurrence, not RTL or target-board evidence",
            },
        },
        "event_control_lane": event_payload,
        "artifacts": {
            "checkpoint": Path(checkpoint_path).as_posix(),
            "checkpoint_sha256": _sha256(checkpoint_target),
            "lookup_checkpoint": LOOKUP_CHECKPOINT.as_posix(),
            "lookup_checkpoint_sha256": _sha256(lookup_path),
            "source_data": Path(source_data_path).as_posix(),
            "source_data_rows": len(source_rows),
        },
        "gate_summary": {"passed": sum(bool(value) for value in gates.values()), "failed": len(failed), "failed_names": failed, "gates": gates},
        "claim_boundary": {
            "allowed": "causal interpretable software exponential recurrence with exact two-cycle assumed-model fidelity, frozen-cutoff transfer, integer-mirror, and separate paired synthetic event-cost comparison",
            "forbidden": "global optimality, physical leakage calibration, paper-number reproduction, logical-error-rate gain from event cost, RTL/synthesis/Fmax, pulse/multilevel/device, or target-board measurement",
        },
    }
    artifact_target = ROOT / Path(artifact_path)
    artifact_target.parent.mkdir(parents=True, exist_ok=True)
    artifact_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    payload = run_exponential_recurrence_baseline(
        artifact_path=args.artifact,
        checkpoint_path=args.checkpoint,
        source_data_path=args.source_data,
    )
    print(json.dumps({"status": payload["status"], "gates": payload["gate_summary"], "comparisons": payload["physical_fidelity_lane"]["comparisons"]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SOURCE_DATA",
    "EventRecurrenceValidationConfig",
    "implementation_sha256",
    "run_exponential_recurrence_baseline",
]
