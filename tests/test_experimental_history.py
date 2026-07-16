from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from cnn_fpga.data.experimental_history import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FORBIDDEN_INPUT_TOKENS,
    UPDATE_STATUSES,
    DeployableLLRContext,
    ExperimentalHistoryBuilder,
    ExperimentalHistoryConfig,
    ExperimentalHistorySample,
    HistoryRuntimeStatus,
    ObservedActionRecord,
    audit_mapping_for_information_leakage,
    runtime_status_from_scheduler,
    schema_provenance,
)
from cnn_fpga.runtime.run_length_fsm import (
    RunLengthFSMInput,
    RunLengthParameterBankFSM,
)
from cnn_fpga.runtime.scheduler import SchedulerEvent
from physics.drift_processes import DriftState
from physics.sbs_observation_reset import PairedSyndrome
from physics.syndrome_stream import ObservedSyndromeStep


def _observed(cycle: int, *, x: str = "g", z: str = "e", run: int = 0) -> ObservedSyndromeStep:
    return ObservedSyndromeStep(
        cycle_index=cycle,
        drift_step=cycle,
        time=float(cycle),
        analog_syndrome=(0.2 + 0.001 * cycle, -0.3),
        residual_syndrome=(0.2, -0.3),
        syndrome=PairedSyndrome(x, z),
        quadrature_phases_rad=(0.0, pi / 2.0),
        x_e_run=run if x == "e" else 0,
        z_e_run=run if z == "e" else 0,
        leakage_run=run if "leakage" in (x, z) else 0,
    )


def _runtime(cycle: int, status: str = "none") -> HistoryRuntimeStatus:
    return HistoryRuntimeStatus(
        cycle_index=cycle,
        fast_deadline_ok=True,
        slow_deadline_ok=status != "stale",
        communication_available=True,
        update_status=status,
        update_applied=status == "committed",
        pending_update=status == "staged",
        active_bank_version=cycle // 3,
        pending_window_count=0,
    )


def _append(builder: ExperimentalHistoryBuilder, cycle: int, **kwargs: object) -> ExperimentalHistorySample:
    observed = kwargs.pop("observed", _observed(cycle))
    action = kwargs.pop("action", ObservedActionRecord.neutral(cycle))
    runtime = kwargs.pop("runtime", _runtime(cycle))
    assert not kwargs
    return builder.append(
        observed,  # type: ignore[arg-type]
        action,  # type: ignore[arg-type]
        DeployableLLRContext((0.35, 0.40)),
        runtime,  # type: ignore[arg-type]
    )


def test_schema_has_all_required_groups_unique_features_and_no_forbidden_name() -> None:
    assert len(FEATURE_NAMES) == 53
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert tuple(FEATURE_GROUPS) == (
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
    )
    assert set(FEATURE_NAMES) == {name for group in FEATURE_GROUPS.values() for name in group}
    normalized = ["".join(character for character in name if character.isalnum()) for name in FEATURE_NAMES]
    assert not any(token in name for token in FORBIDDEN_INPUT_TOKENS for name in normalized)


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: ExperimentalHistoryConfig(history_cycles=1), "at least 2"),
        (lambda: ExperimentalHistoryConfig(llr_clip=0.0), "positive"),
        (lambda: ExperimentalHistoryConfig(run_length_clip=True), "integer"),
        (lambda: DeployableLLRContext((0.0, 1.0)), "positive"),
        (lambda: DeployableLLRContext((1.0, 1.0), source="truth_oracle"), "one of"),
        (lambda: ObservedActionRecord(0, (0.0, 0.0), "normal", 0, False, False, False, "oracle"), "one of"),
        (
            lambda: HistoryRuntimeStatus(0, True, True, True, "bad", False, False, 0, 0),
            "one of",
        ),
        (
            lambda: HistoryRuntimeStatus(0, True, True, True, "none", True, False, 0, 0),
            None,
        ),
    ],
)
def test_config_and_records_fail_closed(factory: object, match: str | None) -> None:
    if match is None:
        assert factory().update_applied is True  # type: ignore[union-attr,operator]
    else:
        with pytest.raises((TypeError, ValueError), match=match):
            factory()  # type: ignore[operator]


@pytest.mark.parametrize("token", FORBIDDEN_INPUT_TOKENS)
def test_leakage_audit_rejects_every_registered_nested_token(token: str) -> None:
    with pytest.raises(ValueError, match="forbidden information token"):
        audit_mapping_for_information_leakage({"safe": [{f"x_{token}_field": 1.0}]})


def test_leakage_audit_rejects_truth_objects_even_without_mapping_key() -> None:
    with pytest.raises(ValueError, match="forbidden truth object"):
        audit_mapping_for_information_leakage([DriftState()])
    with pytest.raises(TypeError, match="unsupported metadata object"):
        audit_mapping_for_information_leakage({"safe": object()})
    with pytest.raises(ValueError, match="forbidden information token"):
        audit_mapping_for_information_leakage({"safe": "hidden_regime=burst"})


def test_runtime_status_uses_real_scheduler_events_with_explicit_priority() -> None:
    events = [
        SchedulerEvent("commit_applied", 7, 35.0, {"version": 2}),
        SchedulerEvent("parameter_update_conflict", 7, 35.0, {"active_version": 2}),
        SchedulerEvent("fast_budget_violation", 7, 35.0, {"latency_us": 2.0}),
    ]
    snapshot = {
        "communication_available": False,
        "pending_windows": 2,
        "param_bank": {"active_version": 2, "pending_commit": None},
    }
    status = runtime_status_from_scheduler(7, events, snapshot, crc_ok=False)

    assert status.update_status == "conflict"
    assert status.update_applied is True
    assert status.fast_deadline_ok is False
    assert status.slow_deadline_ok is True
    assert status.communication_available is False
    assert status.active_bank_version == 2
    assert status.crc_ok is False


def test_runtime_status_rejects_misaligned_or_truth_bearing_scheduler_payload() -> None:
    snapshot = {
        "communication_available": True,
        "pending_windows": 0,
        "param_bank": {"active_version": 0, "pending_commit": None},
    }
    with pytest.raises(ValueError, match="align"):
        runtime_status_from_scheduler(2, [SchedulerEvent("none", 1, 1.0, {})], snapshot)
    with pytest.raises(ValueError, match="hidden"):
        runtime_status_from_scheduler(
            2,
            [SchedulerEvent("window_ready", 2, 1.0, {"payload": {"hidden_regime": "burst"}})],
            snapshot,
        )


def test_action_adapter_consumes_actual_fsm_decision_not_truth() -> None:
    fsm = RunLengthParameterBankFSM()
    decision = fsm.step(
        RunLengthFSMInput(0, (0.2, -0.3), "e", "g", 0)
    )
    action = ObservedActionRecord.from_fsm_decision(decision)

    assert action.cycle_index == 0
    assert action.correction == decision.correction
    assert action.mode == decision.mode
    assert action.parameter_bank_version == decision.parameter_bank_version


def test_first_history_is_left_padded_masked_and_read_only() -> None:
    builder = ExperimentalHistoryBuilder(ExperimentalHistoryConfig(history_cycles=8))
    sample = _append(builder, 0)

    assert sample.values.shape == (8, 53)
    assert np.array_equal(sample.mask, (0, 0, 0, 0, 0, 0, 0, 1))
    assert np.array_equal(sample.cycle_indices, (-1, -1, -1, -1, -1, -1, -1, 0))
    assert np.all(sample.values[:-1] == 0.0)
    assert sample.values.flags.writeable is False
    assert sample.mask.flags.writeable is False
    assert sample.cycle_indices.flags.writeable is False


def test_full_and_rolling_history_have_contiguous_cycle_indices() -> None:
    builder = ExperimentalHistoryBuilder(ExperimentalHistoryConfig(history_cycles=8))
    samples = [_append(builder, cycle) for cycle in range(12)]

    assert np.all(samples[7].mask == 1.0)
    assert np.array_equal(samples[7].cycle_indices, np.arange(8))
    assert np.array_equal(samples[-1].cycle_indices, np.arange(4, 12))
    assert samples[-1].end_cycle == 11


def test_feature_row_encodes_outcome_phase_action_llr_run_and_update_onehots() -> None:
    builder = ExperimentalHistoryBuilder(ExperimentalHistoryConfig(history_cycles=2, run_length_clip=3, llr_clip=0.1))
    observed = _observed(0, x="leakage", z="e", run=9)
    action = ObservedActionRecord(0, (0.4, -0.2), "fallback", 7, True, False, False)
    sample = _append(builder, 0, observed=observed, action=action, runtime=_runtime(0, "committed"))
    row = sample.values[-1]
    lookup = {name: row[index] for index, name in enumerate(FEATURE_NAMES)}

    assert (lookup["syndrome_x_g"], lookup["syndrome_x_e"], lookup["syndrome_x_leakage"]) == (0, 0, 1)
    assert (lookup["syndrome_z_g"], lookup["syndrome_z_e"], lookup["syndrome_z_leakage"]) == (0, 1, 0)
    assert lookup["phase_x_cos"] == pytest.approx(1.0)
    assert lookup["phase_z_sin"] == pytest.approx(1.0)
    assert lookup["action_mode_fallback"] == 1.0
    assert lookup["update_status_committed"] == 1.0
    assert lookup["update_applied"] == 1.0
    assert lookup["leakage_run"] == 3.0
    assert lookup["leakage_run_saturated"] == 1.0
    assert abs(lookup["llr_q"]) <= 0.1 and lookup["llr_q_saturated"] == 1.0


def test_builder_rejects_cycle_gaps_alignment_errors_and_truth_metadata() -> None:
    builder = ExperimentalHistoryBuilder(ExperimentalHistoryConfig(history_cycles=4))
    _append(builder, 0)
    with pytest.raises(ValueError, match="contiguous"):
        _append(builder, 2)
    with pytest.raises(ValueError, match="align"):
        _append(builder, 1, action=ObservedActionRecord.neutral(2))
    with pytest.raises(ValueError, match="logical"):
        builder.append(
            _observed(1),
            ObservedActionRecord.neutral(1),
            DeployableLLRContext((0.4, 0.4)),
            _runtime(1),
            metadata={"logical_truth": "X"},
        )
    bad_scope = replace(_observed(1), observation_scope="simulator_record")
    with pytest.raises(ValueError, match="deployable_observed_syndrome"):
        builder.append(
            bad_scope,
            ObservedActionRecord.neutral(1),
            DeployableLLRContext((0.4, 0.4)),
            _runtime(1),
        )
    with pytest.raises(TypeError, match="cycle_index"):
        builder.append(
            replace(_observed(1), cycle_index=True),
            ObservedActionRecord.neutral(1),
            DeployableLLRContext((0.4, 0.4)),
            _runtime(1),
        )
    with pytest.raises(ValueError, match="at least 0"):
        builder.append(
            replace(_observed(1), leakage_run=-1),
            ObservedActionRecord.neutral(1),
            DeployableLLRContext((0.4, 0.4)),
            _runtime(1),
        )


def test_history_prefix_is_invariant_to_future_appends() -> None:
    builder = ExperimentalHistoryBuilder(ExperimentalHistoryConfig(history_cycles=8))
    prefix = None
    for cycle in range(16):
        sample = _append(builder, cycle)
        if cycle == 7:
            prefix = sample
    assert prefix is not None
    frozen = prefix.values.copy()
    for cycle in range(16, 24):
        _append(builder, cycle)
    np.testing.assert_array_equal(prefix.values, frozen)


def test_sample_constructor_rejects_unmasked_padding() -> None:
    with pytest.raises(ValueError, match="padded rows"):
        ExperimentalHistorySample(
            end_cycle=0,
            values=np.ones((2, len(FEATURE_NAMES))),
            mask=np.asarray((0.0, 1.0)),
            cycle_indices=np.asarray((-1, 0)),
        )


def test_schema_provenance_names_real_producers_and_nonhardware_boundary() -> None:
    provenance = schema_provenance()
    assert provenance["feature_count"] == len(FEATURE_NAMES)
    assert set(provenance["groups"]) == set(FEATURE_GROUPS)
    assert "ObservedSyndromeStep" in provenance["producers"]["syndrome"]
    assert "RunLengthFSMDecision" in provenance["producers"]["action"]
    assert "llr_1d" in provenance["producers"]["llr"]
    assert provenance["hardware_measured"] is False
