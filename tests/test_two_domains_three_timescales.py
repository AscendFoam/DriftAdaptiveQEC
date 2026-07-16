from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from cnn_fpga.runtime.latency_injector import LatencyInjector, StageLatencySpec
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank, PendingCommit
from cnn_fpga.runtime.scheduler import DualLoopScheduler, SchedulerConfig


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "two_domains_three_timescales.json"
DOC_PATH = ROOT / "docs" / "two_domains_three_timescales.md"
HIL_CONFIG_PATH = ROOT / "cnn_fpga" / "config" / "hardware_hil.yaml"


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def constant_injector(*, slow_us: float = 0.0, fast_us: float = 0.0) -> LatencyInjector:
    zero = StageLatencySpec(mean_us=0.0, std_us=0.0, distribution="constant")
    slow = StageLatencySpec(mean_us=slow_us, std_us=0.0, distribution="constant")
    fast = StageLatencySpec(mean_us=fast_us, std_us=0.0, distribution="constant")
    return LatencyInjector(
        dma=zero,
        preprocess=zero,
        inference=slow,
        writeback=zero,
        commit_ack=zero,
        fast_cycle=fast,
        seed=7,
    )


def small_scheduler_config(**overrides: object) -> SchedulerConfig:
    values: dict[str, object] = {
        "t_fast_us": 5.0,
        "window_size": 4,
        "slow_update_period_us": 20.0,
        "window_stride": 4,
        "max_pending_windows": 2,
        "commit_delay_cycles": 1,
        "fast_path_budget_us": 1.5,
        "slow_path_budget_us": 5.0,
        "guard_cycles_after_commit": 0,
    }
    values.update(overrides)
    return SchedulerConfig(**values)


def test_contract_has_exactly_two_domains_and_three_timescales() -> None:
    contract = load_contract()
    assert contract["schema_version"] == "two-domains-three-timescales-v1"
    assert [row["id"] for row in contract["compute_domains"]] == ["CD1", "CD2"]
    assert [row["id"] for row in contract["timescales"]] == ["TS1", "TS2", "TS3"]

    by_scale = {row["id"]: row for row in contract["timescales"]}
    assert by_scale["TS1"]["primary_owner"] == "CD1"
    assert by_scale["TS2"]["primary_owner"] == "CD1"
    assert by_scale["TS2"]["secondary_observer"] == "CD2"
    assert by_scale["TS3"]["primary_owner"] == "CD2"
    assert "not a per-cycle deadline" in by_scale["TS3"]["deadline"]


def test_reference_timing_is_recomputed_from_current_config() -> None:
    contract = load_contract()["reference_timing"]
    config = yaml.safe_load(HIL_CONFIG_PATH.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    timing = config["timing"]

    assert contract["identity"] == "configuration_reference_not_board_measurement"
    assert contract["fast_subcycle_us"] == runtime["t_fast_us"] == 5.0
    assert contract["fast_action_budget_us"] == timing["fast_cycle_budget_us"] == 1.5
    assert contract["window_size_valid_samples"] == runtime["window_size"] == 2048
    assert contract["window_stride_cycles"] == runtime["window_stride"] == 4000
    assert math.isclose(contract["window_content_duration_ms"], 2048 * 5.0 / 1000.0)
    assert math.isclose(contract["window_emission_interval_ms"], 4000 * 5.0 / 1000.0)
    assert contract["slow_start_period_ms"] == runtime["t_slow_update_ms"] == 20.0
    assert contract["slow_job_budget_ms"] == timing["slow_update_budget_us"] / 1000.0 == 5.0
    assert contract["max_pending_windows"] == runtime["max_pending_windows"] == 2
    assert contract["max_bank_age"] is None


def test_cross_domain_interfaces_have_complete_atomic_fields() -> None:
    by_id = {row["id"]: row for row in load_contract()["cross_domain_interfaces"]}
    assert set(by_id) == {"XIF01", "XIF02", "XIF03", "XIF04"}
    assert {"sequence", "n_valid", "payload_length", "CRC"} <= set(by_id["XIF01"]["required_fields"])
    assert {"expected_active_version", "apply_epoch", "CRC", "K", "b"} <= set(
        by_id["XIF02"]["required_fields"]
    )
    assert by_id["XIF03"]["atomicity"] == "compare-and-swap at a cycle boundary"
    assert "readback" in by_id["XIF04"]["atomicity"]


def test_scheduler_commit_occurs_before_fast_callback_with_one_complete_version() -> None:
    bank = ParamBank(DecoderRuntimeParams.identity())
    proposed = DecoderRuntimeParams(K=np.eye(2) * 2.0, b=np.array([0.25, -0.25]))
    pending = bank.stage_update(proposed, commit_epoch=1)
    scheduler = DualLoopScheduler(
        small_scheduler_config(window_size=4),
        param_bank=bank,
        latency_injector=constant_injector(),
    )
    seen: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    events = scheduler.tick_with_fast_path(
        fast_path_fn=lambda epoch, _time, _emit: seen.append(
            (epoch, bank.active_version, bank.read_active().K, bank.read_active().b)
        )
        or None
    )

    assert pending.target_bank == "B"
    assert [(event.kind, event.epoch_id) for event in events] == [("commit_applied", 1)]
    assert seen[0][0:2] == (1, 1)
    np.testing.assert_array_equal(seen[0][2], proposed.K)
    np.testing.assert_array_equal(seen[0][3], proposed.b)
    assert bank.active_bank_name == "B"
    assert bank.has_pending_commit is False


def test_slow_exception_produces_no_stage_and_preserves_active_bank() -> None:
    bank = ParamBank()

    def fail_slow(_window: object, _active: object) -> DecoderRuntimeParams:
        raise RuntimeError("intentional_failure")

    scheduler = DualLoopScheduler(
        small_scheduler_config(window_size=1, window_stride=1, slow_update_period_us=5.0),
        param_bank=bank,
        latency_injector=constant_injector(),
        slow_path_fn=fail_slow,
    )
    scheduler.tick(window_payload={"diagnostics": {"valid_window": True}})
    events = scheduler.tick(window_payload={"diagnostics": {"valid_window": True}})

    assert "slow_update_failed" in [event.kind for event in events]
    assert bank.active_version == 0
    assert bank.active_bank_name == "A"
    assert bank.has_pending_commit is False


def test_bounded_queue_drops_oldest_without_stopping_ticks() -> None:
    scheduler = DualLoopScheduler(
        small_scheduler_config(
            window_size=1,
            window_stride=1,
            slow_update_period_us=5.0,
            max_pending_windows=2,
            slow_path_budget_us=5.0,
        ),
        latency_injector=constant_injector(slow_us=1000.0),
    )
    all_kinds: list[str] = []
    for cycle in range(4):
        events = scheduler.tick(window_payload={"cycle": cycle})
        all_kinds.extend(event.kind for event in events)

    assert scheduler.epoch_id == 4
    assert scheduler.pending_windows == 2
    assert scheduler.dropped_windows == 1
    assert "window_dropped" in all_kinds


def test_fast_deadline_is_honestly_locked_as_record_only_gap() -> None:
    scheduler = DualLoopScheduler(
        small_scheduler_config(fast_path_budget_us=1.0),
        latency_injector=constant_injector(fast_us=2.0),
    )
    called: list[int] = []
    events = scheduler.tick_with_fast_path(
        fast_path_fn=lambda epoch, _time, _emit: called.append(epoch) or None
    )

    assert "fast_budget_violation" in [event.kind for event in events]
    assert called == [1]
    assert load_contract()["current_code_audit"]["fast_deadline_enforcement"] == (
        "record_only_callback_still_executes"
    )


def test_atomic_protocol_exposes_missing_crc_age_ack_and_rollback() -> None:
    protocol = {row["id"]: row for row in load_contract()["atomic_update_protocol"]}
    fields = set(PendingCommit.__dataclass_fields__)

    assert set(protocol) == {f"AP{i:02d}" for i in range(1, 10)}
    assert protocol["AP05"]["current_status"] == "implemented_software"
    assert protocol["AP03"]["current_status"] == "partial_shape_and_finite_only"
    assert protocol["AP08"]["current_status"] == "missing"
    assert not {"schema", "crc", "expected_active_version", "timestamp", "max_age"} & fields


def test_every_failure_branch_has_a_defined_action_and_current_status() -> None:
    rows = load_contract()["failure_matrix"]
    assert [row["id"] for row in rows] == [f"FB{i:02d}" for i in range(1, 15)]
    assert all(row["required_action"].strip() for row in rows)
    assert all(row["current_status"].strip() for row in rows)
    by_id = {row["id"]: row for row in rows}
    assert by_id["FB02"]["current_status"] == "record_only"
    assert "no update" in by_id["FB05"]["required_action"]
    assert "last-known-good" in by_id["FB07"]["required_action"]
    assert "oracle or test lane" in by_id["FB14"]["required_action"]


def test_hidden_truth_is_limited_to_explicit_oracle_modes() -> None:
    audit = load_contract()["current_code_audit"]
    source = (ROOT / "cnn_fpga" / "runtime" / "slow_loop_runtime.py").read_text(encoding="utf-8")
    deployable_must_not = " ".join(load_contract()["compute_domains"][1]["must_not"])

    assert audit["hidden_truth_rule"] == "target_params is benchmark truth; only mock/oracle_delayed may consume it"
    assert 'if self.config.mode in {"mock", "oracle_delayed"}' in source
    assert "return self._predict_from_delayed_oracle(window)" in source
    assert "target_params" in deployable_must_not


def test_human_and_machine_contract_ids_stay_synchronized() -> None:
    contract = load_contract()
    text = DOC_PATH.read_text(encoding="utf-8")

    assert contract["schema_version"] in text
    for section in ("compute_domains", "timescales", "cross_domain_interfaces", "atomic_update_protocol", "failure_matrix"):
        for row in contract[section]:
            assert row["id"] in text
    assert "不晋升" not in text or "CL3" in text
    assert "not a per-cycle deadline" in contract["timescales"][2]["deadline"]
