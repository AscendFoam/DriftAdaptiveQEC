from __future__ import annotations

import pytest

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.atomic_parameter_bank import (
    AtomicParameterBankConfig,
    AtomicParameterBankError,
    serialize_parameter_image,
)
from cnn_fpga.runtime.closed_loop_fault_recovery import (
    MODEL_SCOPE,
    ClosedLoopCycleInput,
    ClosedLoopFaultRecoverySupervisor,
    ClosedLoopRecoveryConfig,
    parameter_image_semantics_sha256,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig


def _image(version: int, profile: int = 0):
    config = ParametricMAPLUTConfig()
    params = registered_parameter_profiles(config)[profile][0]
    return compile_parametric_map_lut(
        params, active_bank_version=version, config=config
    )


def _supervisor(*images):
    config = ClosedLoopRecoveryConfig(
        max_parameter_age_cycles=100,
        host_timeout_cycles=10,
        post_commit_guard_cycles=5,
        guard_blocking_fault_threshold=2,
        ack_timeout_cycles=1,
        transfer_chunk_bytes=17,
    )
    bank_config = AtomicParameterBankConfig(
        min_residency_cycles=1,
        max_payload_age_cycles=100,
    )
    return ClosedLoopFaultRecoverySupervisor(
        images, config=config, bank_config=bank_config
    )


def _cycle(epoch: int, **updates):
    values = {
        "epoch": epoch,
        "syndrome_code": 512,
        "quadrature_phase_bit": epoch % 2,
        "host_heartbeat": True,
    }
    values.update(updates)
    return ClosedLoopCycleInput(**values)


def _prime_and_stage(supervisor, image, *, created: int = 5, apply: int = 6):
    supervisor.observe_selection(window_id=1, selection_key="adaptive", eligible=True)
    supervisor.observe_selection(window_id=2, selection_key="adaptive", eligible=True)
    attempt = supervisor.submit_update(
        image,
        transaction_id=f"tx-v{image.active_bank_version}",
        selection_key="adaptive",
        source_window_id=2,
        created_epoch=created,
        apply_epoch=apply,
    )
    assert attempt.accepted
    return attempt


def test_semantics_hash_ignores_version_but_not_profile() -> None:
    assert parameter_image_semantics_sha256(_image(0, 0)) == parameter_image_semantics_sha256(
        _image(1, 0)
    )
    assert parameter_image_semantics_sha256(_image(0, 0)) != parameter_image_semantics_sha256(
        _image(1, 1)
    )


def test_complete_candidate_commits_on_boundary_and_readback_confirms() -> None:
    supervisor = _supervisor(_image(0), _image(1, 1))
    _prime_and_stage(supervisor, _image(1, 1))

    before = supervisor.tick(_cycle(5))
    committed = supervisor.tick(_cycle(6))

    assert before.active_version == 0 and before.commit_status == "none"
    assert committed.active_version == 1
    assert committed.commit_status == "committed"
    assert committed.readback_status == "confirmed"
    assert committed.map_decision_accepted
    assert committed.conservative_action == "use_validated_map"


def test_lost_ack_keeps_host_uncertain_blocks_next_update_then_readback_recovers() -> None:
    supervisor = _supervisor(_image(0), _image(1, 1), _image(2, 2))
    _prime_and_stage(supervisor, _image(1, 1))
    supervisor.tick(_cycle(5))
    lost = supervisor.tick(
        _cycle(6, host_heartbeat=False, communication_available=False)
    )
    blocked = supervisor.submit_update(
        _image(2, 2),
        transaction_id="tx-v2",
        selection_key="adaptive",
        source_window_id=2,
        created_epoch=6,
        apply_epoch=8,
    )
    waiting = supervisor.tick(
        _cycle(7, host_heartbeat=False, communication_available=False)
    )
    timed_out = supervisor.tick(
        _cycle(8, host_heartbeat=False, communication_available=False)
    )
    recovered = supervisor.tick(_cycle(9, communication_available=True))

    assert lost.readback_status == "awaiting_ack_readback"
    assert lost.active_version == 1
    assert not blocked.accepted and blocked.reason == "ack_readback_pending"
    assert waiting.readback_status == "awaiting_ack_readback"
    assert timed_out.readback_status == "ack_timeout_awaiting_readback"
    assert recovered.readback_status == "confirmed"
    assert recovered.awaiting_readback_version is None


def test_host_timeout_has_defined_frame_hold_and_requests_monotonic_lkg_republish() -> None:
    supervisor = _supervisor(_image(0), _image(1, 0))
    supervisor.tick(_cycle(5))
    for epoch in range(6, 16):
        record = supervisor.tick(_cycle(epoch, host_heartbeat=False))
    timeout = supervisor.tick(_cycle(16, host_heartbeat=False))

    assert timeout.host_timed_out
    assert "deadline_miss" in timeout.fault_flags
    assert timeout.conservative_action == "frame_hold"
    assert not timeout.correction_enable
    assert timeout.recovery_requested
    assert "deadline_miss" in timeout.reason_trace

    attempt = supervisor.submit_lkg_republish(
        _image(1, 0),
        transaction_id="stale-republish",
        selection_key="safe-static",
        evidence_window_ids=(1, 2),
        created_epoch=16,
        apply_epoch=17,
    )
    assert attempt.accepted
    restored = supervisor.tick(_cycle(17, host_heartbeat=True))
    assert restored.active_version == 1
    assert restored.readback_status == "confirmed"
    assert not restored.recovery_requested
    assert restored.active_semantics_sha256 == parameter_image_semantics_sha256(_image(0))


def test_post_commit_integrity_guard_rolls_forward_lkg_contents_not_version() -> None:
    v0 = _image(0, 0)
    v1 = _image(1, 1)
    v2 = _image(2, 0)
    supervisor = _supervisor(v0, v1, v2)
    _prime_and_stage(supervisor, v1)
    supervisor.tick(_cycle(5))
    supervisor.tick(_cycle(6))
    first_fault = supervisor.tick(_cycle(7, reported_integrity_ok=False))
    second_fault = supervisor.tick(_cycle(8, reported_integrity_ok=False))

    assert not first_fault.recovery_requested
    assert second_fault.recovery_requested
    assert {"image_crc_mismatch", "image_sha256_mismatch"} <= set(
        second_fault.fault_flags
    )
    attempt = supervisor.submit_lkg_republish(
        v2,
        transaction_id="guard-republish",
        selection_key="guard-lkg",
        evidence_window_ids=(3, 4),
        created_epoch=8,
        apply_epoch=9,
    )
    assert attempt.accepted
    restored = supervisor.tick(_cycle(9))
    assert restored.active_version == 2
    assert restored.active_semantics_sha256 == parameter_image_semantics_sha256(v0)
    assert restored.active_semantics_sha256 != parameter_image_semantics_sha256(v1)
    assert not restored.recovery_requested


def test_lkg_republish_rejects_wrong_semantics() -> None:
    supervisor = _supervisor(_image(0, 0), _image(1, 1))
    supervisor.tick(_cycle(5))
    for epoch in range(6, 17):
        supervisor.tick(_cycle(epoch, host_heartbeat=False))
    assert supervisor.recovery_requested
    with pytest.raises(AtomicParameterBankError, match="lkg_semantics_mismatch"):
        supervisor.submit_lkg_republish(
            _image(1, 1),
            transaction_id="bad-republish",
            selection_key="safe-static",
            evidence_window_ids=(1, 2),
            created_epoch=16,
            apply_epoch=17,
        )


def test_fresh_confirmed_candidate_clears_stale_recovery_only_after_guard() -> None:
    supervisor = _supervisor(_image(0), _image(1, 1))
    supervisor.tick(_cycle(5))
    for epoch in range(6, 17):
        supervisor.tick(_cycle(epoch, host_heartbeat=False))
    assert supervisor.recovery_requested
    supervisor.observe_selection(window_id=1, selection_key="fresh", eligible=True)
    supervisor.observe_selection(window_id=2, selection_key="fresh", eligible=True)
    attempt = supervisor.submit_update(
        _image(1, 1),
        transaction_id="fresh-after-stale",
        selection_key="fresh",
        source_window_id=2,
        created_epoch=16,
        apply_epoch=17,
    )
    assert attempt.accepted
    committed = supervisor.tick(_cycle(17))
    assert committed.recovery_requested
    for epoch in range(18, 23):
        record = supervisor.tick(_cycle(epoch))
    assert record.recovery_requested
    cleared = supervisor.tick(_cycle(23))
    assert not cleared.recovery_requested
    assert cleared.health_status == "healthy"


def test_corrupt_transfer_never_changes_active_image_or_fast_action_definition() -> None:
    v0 = _image(0)
    v1 = _image(1, 1)
    supervisor = _supervisor(v0, v1)
    supervisor.observe_selection(window_id=1, selection_key="adaptive", eligible=True)
    supervisor.observe_selection(window_id=2, selection_key="adaptive", eligible=True)
    payload = bytearray(serialize_parameter_image(v1))
    payload[len(payload) // 2] ^= 1
    rejected = supervisor.submit_update(
        v1,
        transaction_id="corrupt",
        selection_key="adaptive",
        source_window_id=2,
        created_epoch=5,
        apply_epoch=6,
        payload_override=bytes(payload),
    )
    record = supervisor.tick(_cycle(5))

    assert not rejected.accepted and rejected.reason == "transfer_crc_mismatch"
    assert record.active_version == 0
    assert record.conservative_action in ("use_validated_map", "frame_hold", "reset_request")


def test_submit_rejects_same_version_image_not_registered_in_fast_path() -> None:
    supervisor = _supervisor(_image(0), _image(1, 1))
    supervisor.observe_selection(window_id=1, selection_key="adaptive", eligible=True)
    supervisor.observe_selection(window_id=2, selection_key="adaptive", eligible=True)
    with pytest.raises(AtomicParameterBankError, match="unregistered_fast_path_image"):
        supervisor.submit_update(
            _image(1, 2),
            transaction_id="unregistered",
            selection_key="adaptive",
            source_window_id=2,
            created_epoch=5,
            apply_epoch=6,
        )


def test_leakage_requests_reset_then_returns_to_defined_nonreset_action() -> None:
    supervisor = _supervisor(_image(0))
    records = [supervisor.tick(_cycle(5))]
    records.append(supervisor.tick(_cycle(6, syndrome_x="leakage")))
    records.append(supervisor.tick(_cycle(7, syndrome_x="leakage")))
    records.append(supervisor.tick(_cycle(8, reset_ack=True)))
    records.append(supervisor.tick(_cycle(9)))
    records.append(supervisor.tick(_cycle(10)))

    assert any(row.reset_request for row in records)
    assert records[-1].conservative_action in ("use_validated_map", "frame_hold")
    assert all(row.action_mode for row in records)
    assert all(0 <= row.phase_frame_x_code < 256 for row in records)
    assert all(0 <= row.phase_frame_z_code < 256 for row in records)


def test_cycle_sequence_and_configuration_fail_closed() -> None:
    with pytest.raises(ValueError, match="16-bit"):
        ClosedLoopRecoveryConfig(max_parameter_age_cycles=65536)
    with pytest.raises(ValueError, match="starting at zero"):
        ClosedLoopFaultRecoverySupervisor((_image(1),))
    supervisor = _supervisor(_image(0))
    with pytest.raises(ValueError, match="sequential"):
        supervisor.tick(_cycle(6))
    assert supervisor.config.model_scope == MODEL_SCOPE


def test_closed_loop_supervisor_is_available_through_runtime_public_api() -> None:
    from cnn_fpga.runtime import (
        ClosedLoopFaultRecoverySupervisor as ExportedSupervisor,
        ClosedLoopRecoveryConfig as ExportedConfig,
    )

    assert ExportedSupervisor is ClosedLoopFaultRecoverySupervisor
    assert ExportedConfig is ClosedLoopRecoveryConfig
