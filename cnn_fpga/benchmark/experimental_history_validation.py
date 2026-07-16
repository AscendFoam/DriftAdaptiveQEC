"""T4.1.2 production-style validation for the observed-only history schema.

The workload joins the repository's physical syndrome stream, run-length FSM,
dual-loop scheduler and modular-GKP LLR on the same causal cycle.  Hidden
simulator state is used only to produce observations and is never handed to the
history builder or written to Source Data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from math import pi
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cnn_fpga.data.experimental_history import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FORBIDDEN_INPUT_TOKENS,
    UPDATE_STATUSES,
    DeployableLLRContext,
    ExperimentalHistoryBuilder,
    ExperimentalHistoryConfig,
    ObservedActionRecord,
    audit_mapping_for_information_leakage,
    runtime_status_from_scheduler,
    schema_provenance,
)
from cnn_fpga.runtime.latency_injector import LatencyInjector, StageLatencySpec
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.run_length_fsm import (
    FSM_MODES,
    RunLengthFSMInput,
    RunLengthParameterBankFSM,
)
from cnn_fpga.runtime.scheduler import DualLoopScheduler, SchedulerConfig, WindowFrame
from physics.drift_processes import DriftState
from physics.syndrome_stream import SyndromeStreamConfig, generate_syndrome_stream


@dataclass(frozen=True)
class ExperimentalHistoryValidationConfig:
    seeds: tuple[int, ...] = tuple(range(20261241, 20261249))
    cycles_per_seed: int = 2048
    history_cycles: int = 256
    llr_clip: float = 8.0
    run_length_clip: int = 7
    bank_version_clip: int = 31
    pending_window_clip: int = 2

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        if len(seeds) < 6 or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must contain at least six unique values")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise TypeError("seeds must be nonnegative integers")
        if self.cycles_per_seed < 1024:
            raise ValueError("cycles_per_seed must be at least 1024")
        if self.history_cycles < 32 or self.history_cycles > self.cycles_per_seed:
            raise ValueError("history_cycles must lie in [32, cycles_per_seed]")
        if len(seeds) * self.cycles_per_seed > 100_000:
            raise ValueError("validation workload must not exceed 100,000 cycles")
        for name in ("llr_clip",):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("run_length_clip", "bank_version_clip", "pending_window_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TypeError(f"{name} must be a positive integer")


def _states(cycles: int, seed: int) -> tuple[DriftState, ...]:
    """Four observable stress phases; phase names remain simulator-only."""

    phase = seed % 173
    states: list[DriftState] = []
    for cycle in range(cycles):
        segment = (cycle // 256) % 4
        burst = (cycle + phase) % 193 < 23 or segment == 3 and (cycle % 64) < 16
        sigma_q = (0.21, 0.31, 0.27, 0.39)[segment] * (1.25 if burst else 1.0)
        sigma_p = (0.24, 0.25, 0.36, 0.34)[segment] * (1.20 if burst else 1.0)
        states.append(
            DriftState(
                step=cycle,
                time=float(cycle),
                mu_q=0.055 * np.sin(2.0 * pi * cycle / 509.0),
                mu_p=-0.047 * np.cos(2.0 * pi * cycle / 613.0),
                sigma_q=sigma_q,
                sigma_p=sigma_p,
                rho=(-0.20, 0.42, -0.48, 0.18)[segment],
                loss_gamma=(0.015, 0.035, 0.060, 0.085)[segment],
                p_outlier=min(0.35, (0.01, 0.025, 0.055, 0.10)[segment] + (0.10 if burst else 0.0)),
                outlier_scale=(2.5, 3.0, 3.8, 4.5)[segment],
                burst_active=burst,
                source="registered_synthetic_stress_process",
                regime=f"phase_{segment}",
                seed=seed,
                event_id=cycle // 193 if burst else 0,
            )
        )
    return tuple(states)


def _scheduler(seed: int) -> DualLoopScheduler:
    latency = LatencyInjector(
        dma=StageLatencySpec(24.0, 3.0, min_us=12.0),
        preprocess=StageLatencySpec(28.0, 4.0, min_us=14.0),
        inference=StageLatencySpec(82.0, 14.0, min_us=45.0),
        writeback=StageLatencySpec(18.0, 3.0, min_us=8.0),
        commit_ack=StageLatencySpec(7.0, 1.0, min_us=3.0),
        fast_cycle=StageLatencySpec(1.0, 0.28, min_us=0.2),
        seed=seed + 700_000,
    )

    def slow_path(window: WindowFrame, active: DecoderRuntimeParams) -> DecoderRuntimeParams:
        if window.window_id % 11 == 0:
            raise RuntimeError("registered_injected_slow_path_failure")
        scale = 1.0 - 0.002 * (window.window_id % 7)
        metadata = dict(active.metadata)
        metadata.update(
            {
                "update_family": "registered_observed_window_passthrough",
                "source_window_id": window.window_id,
                "source_window_end_epoch": window.end_epoch,
            }
        )
        return DecoderRuntimeParams(K=active.K * scale, b=active.b.copy(), metadata=metadata)

    return DualLoopScheduler(
        SchedulerConfig(
            t_fast_us=5.0,
            window_size=64,
            window_stride=32,
            slow_update_period_us=80.0,
            max_pending_windows=3,
            commit_delay_cycles=2,
            fast_path_budget_us=1.0,
            slow_path_budget_us=120.0,
            guard_cycles_after_commit=1,
            window_deadline_us=110.0,
        ),
        latency_injector=latency,
        slow_path_fn=slow_path,
    )


def _communication_available(cycle: int, seed: int) -> bool:
    position = (cycle + seed % 211) % 512
    return not 300 <= position < 340


def _crc_ok(cycle: int, seed: int) -> bool:
    return (cycle + seed % 97) % 331 not in (0, 1)


def _normalized(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _implementation_sha256() -> str:
    paths = (
        "cnn_fpga/data/experimental_history.py",
        "cnn_fpga/benchmark/experimental_history_validation.py",
        "cnn_fpga/runtime/run_length_fsm.py",
        "cnn_fpga/runtime/scheduler.py",
        "physics/syndrome_stream.py",
        "physics/ideal_gkp_decoder.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _make_stream(seed: int, cycles: int):
    return generate_syndrome_stream(
        _states(cycles, seed),
        config=SyndromeStreamConfig(
            measurement_sigma=(0.028, 0.031),
            max_recovery_depth=6,
            depth_probability_scale=0.42,
            depth_probability_power=1.7,
            recovery_probability=0.72,
            recovery_gain=0.5,
            base_leakage_probability=0.007,
            loss_leakage_scale=0.025,
            burst_leakage_bonus=0.055,
            higher_leakage_fraction=0.62,
            higher_leakage_mean_duration=5.0,
            readout_fidelity_g=0.985,
            readout_fidelity_e=0.978,
            seed=seed + 300_000,
        ),
    )


def build_experimental_history_validation(
    config: ExperimentalHistoryValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = ExperimentalHistoryValidationConfig() if config is None else config
    if not isinstance(settings, ExperimentalHistoryValidationConfig):
        raise TypeError("config must be ExperimentalHistoryValidationConfig")
    history_config = ExperimentalHistoryConfig(
        history_cycles=settings.history_cycles,
        llr_clip=settings.llr_clip,
        run_length_clip=settings.run_length_clip,
        bank_version_clip=settings.bank_version_clip,
        pending_window_clip=settings.pending_window_clip,
    )
    llr_context = DeployableLLRContext(
        (0.34, 0.36), source="registered_observed_calibration", estimator_version=1
    )
    rows: list[dict[str, object]] = []
    source_digest = hashlib.sha256()
    trace_digests: dict[str, str] = {}
    status_counts = {status: 0 for status in UPDATE_STATUSES}
    mode_counts = {mode: 0 for mode in FSM_MODES}
    outcome_counts = {outcome: 0 for outcome in ("g", "e", "leakage")}
    event_counts: dict[str, int] = {}
    feature_min = np.full(len(FEATURE_NAMES), np.inf)
    feature_max = np.full(len(FEATURE_NAMES), -np.inf)
    prefix_immutable = True
    leakage_probe_rejected = False

    for seed in settings.seeds:
        stream = _make_stream(seed, settings.cycles_per_seed)
        scheduler = _scheduler(seed)
        fsm = RunLengthParameterBankFSM()
        builder = ExperimentalHistoryBuilder(history_config)
        trace_digest = hashlib.sha256()
        frozen_prefix = None
        frozen_prefix_values = None

        for cycle, step in enumerate(stream.steps):
            observed = step.observed
            events = []
            if cycle > 0:
                events.extend(
                    scheduler.tick(
                        window_payload=observed.as_deployable_dict(),
                        communication_available=_communication_available(cycle, seed),
                    )
                )
                if cycle % 389 in (70, 71):
                    params = DecoderRuntimeParams.identity()
                    params.metadata = {
                        "update_family": "registered_external_probe",
                        "source_cycle": cycle,
                    }
                    _, external_events = scheduler.stage_external_update(
                        params,
                        commit_epoch=cycle + 25,
                        metadata={"update_family": "registered_external_probe"},
                    )
                    events.extend(external_events)
            snapshot = scheduler.snapshot()
            crc_ok = _crc_ok(cycle, seed)
            runtime = runtime_status_from_scheduler(cycle, events, snapshot, crc_ok=crc_ok)
            fsm_input = RunLengthFSMInput(
                cycle_index=cycle,
                residual=observed.residual_syndrome,
                syndrome_x=observed.syndrome.x,
                syndrome_z=observed.syndrome.z,
                quadrature_phase_bit=cycle & 1,
                valid=observed.valid,
                crc_ok=crc_ok,
                parameter_fresh=runtime.communication_available
                and runtime.update_status not in {"failed", "stale"},
                deadline_ok=runtime.fast_deadline_ok and runtime.slow_deadline_ok,
            )
            action = ObservedActionRecord.from_fsm_decision(fsm.step(fsm_input))
            sample = builder.append(
                observed,
                action,
                llr_context,
                runtime,
                metadata={"source_seed": seed, "source_cycle": cycle},
            )
            row_values = sample.values[-1]
            feature_min = np.minimum(feature_min, row_values)
            feature_max = np.maximum(feature_max, row_values)
            trace_digest.update(np.asarray(row_values, dtype="<f8").tobytes())
            trace_digest.update(np.asarray(sample.mask, dtype="<f8").tobytes())

            status_counts[runtime.update_status] += 1
            mode_counts[action.mode] += 1
            outcome_counts[observed.syndrome.x] += 1
            outcome_counts[observed.syndrome.z] += 1
            for event in events:
                event_counts[event.kind] = event_counts.get(event.kind, 0) + 1

            source_row: dict[str, object] = {
                "seed": seed,
                "cycle": cycle,
                "history_valid_cycles": int(np.sum(sample.mask)),
                "history_start_cycle": int(sample.cycle_indices[sample.mask == 1.0][0]),
                "history_end_cycle": sample.end_cycle,
                "update_status": runtime.update_status,
                "action_mode": action.mode,
                "scheduler_event_kinds": ";".join(event.kind for event in events) or "none",
            }
            source_row.update(
                {name: float(row_values[index]) for index, name in enumerate(FEATURE_NAMES)}
            )
            rows.append(source_row)
            source_digest.update(
                (json.dumps(source_row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            )

            if cycle == settings.history_cycles - 1:
                frozen_prefix = sample
                frozen_prefix_values = sample.values.copy()

        if frozen_prefix is None or frozen_prefix_values is None:
            raise RuntimeError("validation failed to capture a full prefix")
        prefix_immutable &= bool(np.array_equal(frozen_prefix.values, frozen_prefix_values))
        trace_digests[str(seed)] = trace_digest.hexdigest()

        if seed == settings.seeds[0]:
            probe_builder = ExperimentalHistoryBuilder(history_config)
            try:
                probe_builder.append(
                    stream.steps[0].observed,
                    ObservedActionRecord.neutral(0),
                    llr_context,
                    runtime_status_from_scheduler(0, [], _scheduler(seed).snapshot()),
                    metadata={"safe_container": stream.steps[0].truth},
                )
            except ValueError:
                leakage_probe_rejected = True

    source_fields = tuple(rows[0]) if rows else ()
    input_names = tuple(FEATURE_NAMES)
    normalized_inputs = tuple(_normalized(name) for name in input_names)
    forbidden_input_hits = sorted(
        {
            token
            for token in FORBIDDEN_INPUT_TOKENS
            for name in normalized_inputs
            if token in name
        }
    )
    expected_rows = len(settings.seeds) * settings.cycles_per_seed
    one_hot_groups = {
        "syndrome_x": ("syndrome_x_g", "syndrome_x_e", "syndrome_x_leakage"),
        "syndrome_z": ("syndrome_z_g", "syndrome_z_e", "syndrome_z_leakage"),
        "action_mode": tuple(f"action_mode_{mode}" for mode in FSM_MODES),
        "update_status": tuple(f"update_status_{status}" for status in UPDATE_STATUSES),
    }
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    one_hot_valid = True
    for source_row in rows:
        for names in one_hot_groups.values():
            one_hot_valid &= abs(sum(float(source_row[name]) for name in names) - 1.0) < 1.0e-12
    gates = {
        "source_grid_is_complete": len(rows) == expected_rows,
        "all_seed_traces_are_unique": len(set(trace_digests.values())) == len(settings.seeds),
        "schema_has_required_groups_and_53_unique_features": (
            len(FEATURE_NAMES) == 53
            and len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
            and set(FEATURE_GROUPS)
            == {
                "analog_syndrome",
                "residual_syndrome",
                "observed_outcome",
                "quadrature_phase",
                "recent_action",
                "soft_information",
                "run_length",
                "deadline_health",
                "parameter_update",
                "record_health",
            }
        ),
        "input_schema_has_no_registered_forbidden_token": not forbidden_input_hits,
        "truth_object_negative_probe_is_rejected": leakage_probe_rejected,
        "source_data_has_no_truth_or_hidden_field": not any(
            token in _normalized(field)
            for field in source_fields
            for token in FORBIDDEN_INPUT_TOKENS
        ),
        "all_source_features_are_finite": bool(
            np.all(np.isfinite(feature_min)) and np.all(np.isfinite(feature_max))
        ),
        "all_categorical_groups_are_exactly_one_hot": one_hot_valid,
        "left_padding_and_full_history_are_exercised": (
            any(int(row["history_valid_cycles"]) == 1 for row in rows)
            and any(int(row["history_valid_cycles"]) == settings.history_cycles for row in rows)
            and all(
                int(row["history_start_cycle"])
                == max(0, int(row["cycle"]) - settings.history_cycles + 1)
                for row in rows
            )
        ),
        "captured_prefix_is_immutable_after_future_appends": prefix_immutable,
        "all_observed_outcomes_are_exercised": min(outcome_counts.values()) > 0,
        "all_real_fsm_modes_are_exercised": min(mode_counts.values()) > 0,
        "all_update_statuses_are_exercised": min(status_counts.values()) > 0,
        "deadline_pause_crc_and_failure_paths_are_exercised": (
            event_counts.get("fast_budget_violation", 0) > 0
            and event_counts.get("slow_budget_violation", 0) > 0
            and event_counts.get("window_deadline_miss", 0) > 0
            and event_counts.get("communication_pause_started", 0) > 0
            and event_counts.get("slow_update_failed", 0) > 0
            and feature_min[feature_index["crc_ok"]] == 0.0
        ),
        "all_saturation_paths_are_exercised": all(
            feature_max[feature_index[name]] == 1.0
            for name in (
                "llr_q_saturated",
                "llr_p_saturated",
                "x_e_run_saturated",
                "z_e_run_saturated",
                "leakage_run_saturated",
                "active_bank_version_saturated",
                "pending_window_count_saturated",
            )
        ),
        "scheduler_status_is_derived_not_hand_labeled": (
            event_counts.get("commit_applied", 0) > 0
            and event_counts.get("external_params_staged", 0) > 0
            and event_counts.get("parameter_update_conflict", 0) > 0
        ),
        "scope_is_synthetic_software_not_device_measurement": schema_provenance()["hardware_measured"] is False,
    }
    gates = {name: bool(passed) for name, passed in gates.items()}
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t4.1.2-experimental-history-validation-v1",
        "task_id": "T4.1.2",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "source_rows_sha256": source_digest.hexdigest(),
        "validation_config": asdict(settings),
        "history_schema": schema_provenance(),
        "causal_contract": {
            "alignment": "post-cycle observed syndrome, applied fast action and scheduler status at cycle t; no future sample is visible",
            "padding": "left zero padding is excluded by an explicit binary mask",
            "llr_provenance": "fixed registered observed calibration, never simulator state",
            "scheduler_provenance": "DualLoopScheduler events and snapshot, including failures, stale windows, conflicts and commits",
            "truth_use": "truth exists only inside the synthetic producer and the negative rejection probe",
        },
        "aggregate": {
            "seeds": len(settings.seeds),
            "cycles": expected_rows,
            "source_data_rows": len(rows),
            "feature_count": len(FEATURE_NAMES),
            "trace_sha256": trace_digests,
            "update_status_counts": status_counts,
            "action_mode_counts": mode_counts,
            "observed_outcome_counts": outcome_counts,
            "scheduler_event_counts": dict(sorted(event_counts.items())),
            "feature_ranges": {
                name: {"min": float(feature_min[index]), "max": float(feature_max[index])}
                for index, name in enumerate(FEATURE_NAMES)
            },
        },
        "information_leakage_audit": {
            "registered_forbidden_tokens": list(FORBIDDEN_INPUT_TOKENS),
            "input_forbidden_hits": forbidden_input_hits,
            "source_fields": list(source_fields),
            "truth_object_negative_probe_rejected": leakage_probe_rejected,
        },
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": "causal observed-only 256-cycle software history schema with real repository producers and explicit leakage rejection",
            "forbidden": "device-calibrated IQ/ADC semantics, target-board timing, learned-model gain, logical-error-rate gain, or hidden-state availability",
        },
    }
    return payload, rows


def write_experimental_history_validation(
    json_path: str | Path = "docs/t4_1_2_experimental_history_validation.json",
    csv_path: str | Path = "docs/t4_1_2_experimental_history_source_data.csv",
    config: ExperimentalHistoryValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_experimental_history_validation(config)
    if not rows:
        raise RuntimeError("experimental history validation produced no Source Data")
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
    parser.add_argument("--json", default="docs/t4_1_2_experimental_history_validation.json")
    parser.add_argument("--csv", default="docs/t4_1_2_experimental_history_source_data.csv")
    args = parser.parse_args(argv)
    payload = write_experimental_history_validation(args.json, args.csv)
    print(json.dumps({"status": payload["status"], "gate_summary": payload["gate_summary"]}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExperimentalHistoryValidationConfig",
    "build_experimental_history_validation",
    "write_experimental_history_validation",
]
