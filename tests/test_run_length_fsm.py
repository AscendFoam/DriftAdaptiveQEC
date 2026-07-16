from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from cnn_fpga.runtime.run_length_fsm import (
    FALLBACK,
    FSM_MODES,
    LEAKAGE_HOLD,
    NORMAL,
    X_RECOVERY,
    Z_RECOVERY,
    RunLengthFSMConfig,
    RunLengthFSMInput,
    RunLengthParameterBankFSM,
    RunLengthParameterTable,
)


def _event(
    cycle: int,
    *,
    x: str = "g",
    z: str = "g",
    phase: int = 0,
    residual: tuple[float, float] = (0.4, -0.2),
    **health: bool,
) -> RunLengthFSMInput:
    return RunLengthFSMInput(
        cycle_index=cycle,
        residual=residual,
        syndrome_x=x,
        syndrome_z=z,
        quadrature_phase_bit=phase,
        **health,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RunLengthFSMConfig(counter_bits=True),
        lambda: RunLengthFSMConfig(counter_bits=1),
        lambda: RunLengthFSMConfig(counter_bits=17),
        lambda: RunLengthFSMConfig(e_enter_run=0),
        lambda: RunLengthFSMConfig(leakage_enter_run=8),
        lambda: RunLengthFSMConfig(leakage_clear_run=0),
        lambda: RunLengthFSMConfig(fallback_clear_run=0),
        lambda: RunLengthFSMConfig(correction_limit=float("nan")),
        lambda: RunLengthFSMConfig(correction_limit=0.0),
        lambda: _event(0, phase=2),
        lambda: _event(0, x="unknown"),
        lambda: _event(0, residual=(float("inf"), 0.0)),
        lambda: RunLengthParameterBankFSM(config=object()),
        lambda: RunLengthParameterBankFSM(parameter_table=object()),
        lambda: RunLengthParameterBankFSM(param_bank=object()),
    ],
)
def test_contract_fails_closed(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_parameter_table_is_exact_and_copy_isolated() -> None:
    table = RunLengthParameterTable()
    assert tuple(FSM_MODES) == (
        NORMAL,
        X_RECOVERY,
        Z_RECOVERY,
        LEAKAGE_HOLD,
        FALLBACK,
    )
    first = table.params(X_RECOVERY)
    first.K[0, 0] = 99.0
    np.testing.assert_array_equal(table.params(X_RECOVERY).K, np.diag([0.65, 1.0]))
    with pytest.raises(ValueError, match="exactly"):
        RunLengthParameterTable({NORMAL: DecoderRuntimeParams.identity()})


def test_e_run_threshold_and_phase_bit_break_simultaneous_tie() -> None:
    fsm = RunLengthParameterBankFSM(RunLengthFSMConfig(e_enter_run=2))
    first = fsm.step(_event(0, x="e", z="e", phase=0))
    second = fsm.step(_event(1, x="e", z="e", phase=1))

    assert first.mode == NORMAL
    assert (first.x_e_run, first.z_e_run) == (1, 1)
    assert second.mode == Z_RECOVERY
    assert second.requested_mode == Z_RECOVERY
    assert second.reason == "both_e_runs_phase_z_priority"
    assert second.bank_switched is True
    np.testing.assert_allclose(second.correction, (0.4, -0.13))


def test_leakage_has_configurable_entry_and_clear_hysteresis() -> None:
    fsm = RunLengthParameterBankFSM(
        RunLengthFSMConfig(leakage_enter_run=2, leakage_clear_run=2)
    )
    decisions = [
        fsm.step(_event(0, x="leakage")),
        fsm.step(_event(1, z="leakage")),
        fsm.step(_event(2)),
        fsm.step(_event(3)),
    ]

    assert [decision.mode for decision in decisions] == [
        NORMAL,
        LEAKAGE_HOLD,
        LEAKAGE_HOLD,
        NORMAL,
    ]
    assert decisions[1].correction == (0.0, 0.0)
    assert decisions[2].reason == "leakage_clear_hysteresis"
    assert decisions[3].leakage_clean_run == 2


def test_health_fault_has_priority_and_fallback_clear_is_hysteretic() -> None:
    fsm = RunLengthParameterBankFSM(RunLengthFSMConfig(fallback_clear_run=2))
    decisions = [
        fsm.step(_event(0, x="leakage", deadline_ok=False)),
        fsm.step(_event(1)),
        fsm.step(_event(2)),
    ]

    assert [decision.mode for decision in decisions] == [FALLBACK, FALLBACK, NORMAL]
    assert decisions[0].reason == "health_fault:deadline"
    assert decisions[1].reason == "fallback_clear_hysteresis"
    assert decisions[0].correction == pytest.approx((0.2, -0.1))
    assert fsm.state.fallback_count == 2


def test_three_bit_run_counters_saturate_without_wrapping() -> None:
    fsm = RunLengthParameterBankFSM(RunLengthFSMConfig(counter_bits=3, e_enter_run=2))
    decisions = [fsm.step(_event(cycle, x="e")) for cycle in range(20)]
    assert decisions[-1].x_e_run == 7
    assert decisions[-1].z_e_run == 0
    assert fsm.state.x_e_run == 7
    assert all(decision.x_e_run <= 7 for decision in decisions)


def test_real_parameter_bank_switches_once_per_required_mode_sync() -> None:
    fsm = RunLengthParameterBankFSM(RunLengthFSMConfig(e_enter_run=2))
    normal_0 = fsm.step(_event(0))
    normal_1 = fsm.step(_event(1))
    e_1 = fsm.step(_event(2, x="e"))
    recovery = fsm.step(_event(3, x="e"))
    recovery_hold = fsm.step(_event(4, x="e"))
    normal_2 = fsm.step(_event(5))

    assert [normal_0.parameter_bank_version, normal_1.parameter_bank_version, e_1.parameter_bank_version] == [0, 0, 0]
    assert recovery.parameter_bank_version == 1
    assert recovery.bank_switched is True
    assert recovery_hold.parameter_bank_version == 1
    assert recovery_hold.bank_switched is False
    assert normal_2.parameter_bank_version == 2
    assert fsm.state.transition_count == 2
    assert fsm.param_bank.read_active().metadata["mode"] == NORMAL


def test_pending_external_writer_keeps_every_cycle_in_local_safe_rom_until_sync() -> None:
    table = RunLengthParameterTable()
    bank = ParamBank(table.params(NORMAL))
    external = DecoderRuntimeParams(
        K=np.diag([9.0, 9.0]),
        b=np.zeros(2),
        metadata={"mode": "external_untrusted"},
    )
    pending = bank.stage_update(external, commit_epoch=3, metadata={"writer": "external"})
    fsm = RunLengthParameterBankFSM(
        RunLengthFSMConfig(e_enter_run=1, fallback_clear_run=1),
        parameter_table=table,
        param_bank=bank,
    )

    conflicts = [fsm.step(_event(cycle, x="e")) for cycle in range(3)]
    assert all(decision.mode == FALLBACK for decision in conflicts)
    assert all(decision.requested_mode == X_RECOVERY for decision in conflicts)
    assert all(decision.bank_conflict for decision in conflicts)
    assert all(decision.local_safe_rom_used for decision in conflicts)
    assert all(decision.correction == pytest.approx((0.2, -0.1)) for decision in conflicts)
    assert bank.snapshot()["pending_commit"]["version"] == pending.version

    synchronized = fsm.step(_event(3, x="e"))
    assert synchronized.mode == X_RECOVERY
    assert synchronized.bank_conflict is False
    assert synchronized.bank_switched is True
    assert synchronized.parameter_bank_version == 2
    assert bank.has_pending_commit is False
    assert bank.read_active().metadata["mode"] == X_RECOVERY
    np.testing.assert_allclose(synchronized.correction, (0.26, -0.2))


def test_output_correction_is_bounded_and_uses_active_mode_parameters() -> None:
    fsm = RunLengthParameterBankFSM(
        RunLengthFSMConfig(e_enter_run=1, correction_limit=0.3)
    )
    decision = fsm.step(_event(0, x="e", residual=(2.0, -2.0)))
    assert decision.mode == X_RECOVERY
    assert decision.correction == pytest.approx((0.3, -0.3))


def test_replay_and_gap_rejection_are_transactional() -> None:
    fsm = RunLengthParameterBankFSM()
    initial_state = fsm.state
    initial_bank = fsm.param_bank.snapshot()
    with pytest.raises(ValueError, match="sequential"):
        fsm.step(_event(1))
    assert fsm.state == initial_state
    assert fsm.param_bank.snapshot() == initial_bank

    fsm.step(_event(0))
    state_after = fsm.state
    bank_after = fsm.param_bank.snapshot()
    with pytest.raises(ValueError, match="sequential"):
        fsm.step(_event(0))
    assert fsm.state == state_after
    assert fsm.param_bank.snapshot() == bank_after


def test_online_input_schema_contains_no_hidden_truth_or_regime() -> None:
    names = {field.name for field in fields(RunLengthFSMInput)}
    assert names == {
        "cycle_index",
        "residual",
        "syndrome_x",
        "syndrome_z",
        "quadrature_phase_bit",
        "valid",
        "crc_ok",
        "parameter_fresh",
        "deadline_ok",
    }
    assert not any("truth" in name or "hidden" in name or "regime" in name for name in names)

