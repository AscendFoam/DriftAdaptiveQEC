from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.runtime.latency_injector import LatencyInjector, StageLatencySpec
from cnn_fpga.runtime.param_bank import (
    DecoderRuntimeParams,
    ParamBank,
    ParameterUpdateConflictError,
)
from cnn_fpga.runtime.scheduler import DualLoopScheduler, SchedulerConfig
from cnn_fpga.runtime.timing_fault_model import (
    DEFAULT_CONFIG,
    ROOT,
    TimingFaultScenario,
    TimingStressConfig,
    _sha256_sources,
    default_scenarios,
    simulate_scenario,
)
from cnn_fpga.utils.config import load_yaml_config


ARTIFACT = ROOT / "docs" / "t2_4_2_timing_fault_validation.json"


def _constant_injector(*, slow_us: float = 0.0, fast_us: float = 0.0) -> LatencyInjector:
    zero = StageLatencySpec(mean_us=0.0, distribution="constant")
    return LatencyInjector(
        dma=zero,
        preprocess=zero,
        inference=StageLatencySpec(mean_us=slow_us, distribution="constant"),
        writeback=zero,
        commit_ack=zero,
        fast_cycle=StageLatencySpec(mean_us=fast_us, distribution="constant"),
        seed=242,
    )


def _small_scheduler(**updates: object) -> SchedulerConfig:
    values: dict[str, object] = {
        "t_fast_us": 5.0,
        "window_size": 1,
        "slow_update_period_us": 5.0,
        "window_stride": 100,
        "max_pending_windows": 2,
        "commit_delay_cycles": 1,
        "fast_path_budget_us": 1.5,
        "slow_path_budget_us": 5.0,
        "window_deadline_us": 10.0,
    }
    values.update(updates)
    return SchedulerConfig(**values)


def test_parameter_bank_rejects_second_writer_without_mutating_pending_state() -> None:
    bank = ParamBank()
    first = DecoderRuntimeParams(K=np.eye(2), b=np.asarray([0.2, 0.0]))
    second = DecoderRuntimeParams(K=np.eye(2), b=np.asarray([9.0, 0.0]))
    pending = bank.stage_update(first, commit_epoch=5, metadata={"writer": "slow"})
    snapshot_before = bank.snapshot()

    with pytest.raises(ParameterUpdateConflictError, match="pending version"):
        bank.stage_update(second, commit_epoch=2, metadata={"writer": "external"})

    assert bank.snapshot() == snapshot_before
    assert bank.has_pending_commit is True
    assert bank.snapshot()["pending_commit"]["version"] == pending.version
    assert bank.active_version == 0
    bank.commit_if_ready(5)
    assert bank.active_version == 1
    np.testing.assert_array_equal(bank.read_active().b, first.b)


def test_input_burst_emits_explicit_fifo_overflow_and_drop_oldest_provenance() -> None:
    scheduler = DualLoopScheduler(
        _small_scheduler(max_pending_windows=2),
        latency_injector=_constant_injector(),
    )
    events = scheduler.inject_window_burst([{"member": index} for index in range(4)])
    kinds = [event.kind for event in events]

    assert kinds.count("input_burst") == 1
    assert kinds.count("window_ready") == 4
    assert kinds.count("fifo_overflow") == 2
    assert kinds.count("window_dropped") == 2
    assert scheduler.pending_windows == 2
    assert scheduler.snapshot()["fifo_overflows"] == 2
    dropped_ids = [
        event.details["dropped_window_id"]
        for event in events
        if event.kind == "fifo_overflow"
    ]
    assert dropped_ids == [1, 2]
    assert all(
        event.details["reason"] == "fifo_overflow_drop_oldest"
        for event in events
        if event.kind == "window_dropped"
    )


def test_communication_pause_is_timed_and_enters_end_to_end_window_deadline() -> None:
    scheduler = DualLoopScheduler(
        _small_scheduler(),
        latency_injector=_constant_injector(slow_us=10.0),
    )
    scheduler.tick(window_payload={"observed_mean": 0.1, "n_valid": 1})
    pause_start = scheduler.tick(communication_available=False)
    scheduler.tick(communication_available=False)
    resume = scheduler.tick(communication_available=True)

    assert [event.kind for event in pause_start] == ["communication_pause_started"]
    resume_by_kind = {event.kind: event for event in resume}
    assert {
        "communication_pause_ended",
        "window_deadline_miss",
        "slow_update_finished",
        "params_staged",
    } <= set(resume_by_kind)
    pause = resume_by_kind["communication_pause_ended"]
    assert pause.details["duration_cycles"] == 2
    assert pause.details["duration_us"] == pytest.approx(10.0)
    miss = resume_by_kind["window_deadline_miss"]
    assert miss.details["service_latency_us"] == pytest.approx(10.0)
    assert miss.details["window_age_us"] == pytest.approx(15.0)
    assert miss.details["window_age_us"] > miss.details["deadline_us"]
    assert scheduler.snapshot()["communication_paused_cycles"] == 2


def test_external_parameter_conflict_has_structured_fail_closed_event() -> None:
    scheduler = DualLoopScheduler(
        _small_scheduler(),
        latency_injector=_constant_injector(),
    )
    first = DecoderRuntimeParams(K=np.eye(2), b=np.asarray([0.3, 0.0]))
    conflicting = DecoderRuntimeParams(K=np.eye(2), b=np.asarray([7.0, 0.0]))
    staged, first_events = scheduler.stage_external_update(first, commit_epoch=2)
    rejected, conflict_events = scheduler.stage_external_update(conflicting, commit_epoch=1)

    assert staged is not None
    assert [event.kind for event in first_events] == ["external_params_staged"]
    assert rejected is None
    assert [event.kind for event in conflict_events] == ["parameter_update_conflict"]
    assert scheduler.param_bank.has_pending_commit is True
    assert scheduler.param_bank.snapshot()["pending_commit"]["version"] == staged.version
    assert scheduler.snapshot()["parameter_update_conflicts"] == 1
    scheduler.tick()
    scheduler.tick()
    np.testing.assert_array_equal(scheduler.param_bank.read_active().b, first.b)


def test_fast_latency_is_exposed_without_changing_record_only_runtime_contract() -> None:
    scheduler = DualLoopScheduler(
        _small_scheduler(fast_path_budget_us=1.0),
        latency_injector=_constant_injector(fast_us=2.0),
    )
    called: list[int] = []
    events = scheduler.tick_with_fast_path(
        fast_path_fn=lambda epoch, _time, _emit: called.append(epoch) or None
    )
    assert scheduler.last_fast_cycle_latency_us == pytest.approx(2.0)
    assert scheduler.snapshot()["last_fast_cycle_latency_us"] == pytest.approx(2.0)
    assert "fast_budget_violation" in [event.kind for event in events]
    assert called == [1]


def test_single_scenario_simulation_is_seed_deterministic_and_uses_live_config() -> None:
    source = deepcopy(load_yaml_config(DEFAULT_CONFIG))
    source["runtime"].update(
        {
            "window_size": 32,
            "window_stride": 64,
            "t_slow_update_ms": 0.32,
            "max_pending_windows": 2,
        }
    )
    source["timing"].update(
        {"fast_cycle_budget_us": 1.5, "slow_update_budget_us": 5000.0}
    )
    cfg = TimingStressConfig(
        n_cycles=1024,
        seeds=(13, 17),
        bootstrap_replicates=1000,
    )
    scenario = TimingFaultScenario(name="reference")
    first = simulate_scenario(scenario, config=cfg, seed=13, yaml_config=source)
    second = simulate_scenario(scenario, config=cfg, seed=13, yaml_config=source)

    assert first == second
    assert first["n_cycles"] == 1024
    assert first["scheduler_snapshot"]["window_stride"] == 64
    assert first["integrity"]["slow_estimator_uses_hidden_truth"] is False
    assert first["target_hardware_measured"] is False


def test_default_scenarios_reject_runs_too_short_to_contain_faults() -> None:
    with pytest.raises(ValueError, match="at least 24,000 cycles"):
        default_scenarios(10_000)


def test_production_artifact_has_per_seed_detection_effects_and_live_source_hash() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["contract_id"] == "T242-PAIRED-SCHEDULER-TIMING-FAULT-V1"
    assert artifact["status"] == "PASS"
    assert all(artifact["gates"].values())
    assert artifact["target_hardware_measured"] is False
    assert artifact["config"]["n_cycles"] >= 64_000
    assert len(artifact["config"]["seeds"]) >= 8

    expected_names = [scenario.name for scenario in default_scenarios(64_000)]
    assert [row["name"] for row in artifact["scenarios"]] == expected_names
    assert len(artifact["per_seed_results"]) == len(expected_names) * 8

    required_events = {
        "jitter_deadline": {
            "fast_budget_violation",
            "slow_budget_violation",
            "window_deadline_miss",
        },
        "input_burst": {"input_burst"},
        "communication_pause": {
            "communication_pause_started",
            "communication_pause_ended",
            "window_deadline_miss",
        },
        "parameter_conflict": {"parameter_update_conflict"},
        "fifo_overflow": {"input_burst", "fifo_overflow", "window_dropped"},
        "combined": {
            "fast_budget_violation",
            "slow_budget_violation",
            "input_burst",
            "communication_pause_started",
            "communication_pause_ended",
            "parameter_update_conflict",
            "fifo_overflow",
            "window_deadline_miss",
        },
    }
    for row in artifact["per_seed_results"]:
        assert row["integrity"]["active_version_monotonic"] is True
        assert row["integrity"]["external_conflicting_updates_applied"] == 0
        for event in required_events.get(row["scenario"], set()):
            assert row["event_counts"].get(event, 0) > 0

    aggregate = {row["scenario"]: row for row in artifact["aggregates"]}
    for row in aggregate.values():
        assert "fast_action_availability" in row
        assert "fresh_parameter_availability" in row
        assert "end_to_end_control_availability" in row
    assert aggregate["combined"]["paired_ler_minus_reference"]["ci_low"] > 0.0
    assert aggregate["combined"]["paired_availability_minus_reference"]["ci_high"] < 0.0

    expected_hash = _sha256_sources(
        [
            ROOT / "cnn_fpga" / "runtime" / "timing_fault_model.py",
            ROOT / "cnn_fpga" / "runtime" / "scheduler.py",
            ROOT / "cnn_fpga" / "runtime" / "param_bank.py",
        ]
    )
    assert artifact["implementation_sha256"] == expected_hash
    assert "board" in " ".join(artifact["claim_boundary"]["forbidden"]).lower()
