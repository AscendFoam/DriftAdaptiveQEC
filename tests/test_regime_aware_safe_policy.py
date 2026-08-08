from __future__ import annotations

from dataclasses import replace

import pytest

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.atomic_parameter_bank import AtomicParameterBankConfig
from cnn_fpga.runtime.closed_loop_fault_recovery import (
    ClosedLoopCycleInput,
    ClosedLoopRecoveryConfig,
    parameter_image_semantics_sha256,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig
from cnn_fpga.runtime.regime_aware_safe_policy import (
    ADAPTIVE_OPEN,
    INTEGRITY_ROLLBACK,
    LEAKAGE_RESET,
    POSTERIOR_UNCERTAIN,
    RECOVERING,
    TAIL_TRUSTED,
    AdaptiveMAPCandidate,
    ObservedRegimePosterior,
    RegimeAwareSafeAdaptivePolicy,
    RouteACycleInput,
    RouteAPolicyConfig,
    clone_parameter_image,
)


MODEL_HASH = "4" * 64


def _image(profile: int = 0, version: int = 0):
    config = ParametricMAPLUTConfig()
    params = registered_parameter_profiles(config)[profile][0]
    return compile_parametric_map_lut(
        params, active_bank_version=version, config=config
    )


def _policy() -> RegimeAwareSafeAdaptivePolicy:
    config = RouteAPolicyConfig(
        regime_window_cycles=4,
        parameter_update_period_cycles=16,
        recovery_hysteresis_windows=2,
        emergency_confirmation_events=2,
        max_parameter_age_cycles=100,
    )
    recovery = ClosedLoopRecoveryConfig(
        max_parameter_age_cycles=100,
        host_timeout_cycles=100,
        post_commit_guard_cycles=4,
        guard_blocking_fault_threshold=2,
        ack_timeout_cycles=2,
        transfer_chunk_bytes=17,
    )
    bank = AtomicParameterBankConfig(
        promotion_good_windows=2,
        min_residency_cycles=1,
        max_payload_age_cycles=100,
    )
    return RegimeAwareSafeAdaptivePolicy(
        _image(0), config=config, recovery_config=recovery, bank_config=bank
    )


def _posterior(
    available: int,
    probabilities=(0.55, 0.35, 0.05, 0.05),
) -> ObservedRegimePosterior:
    return ObservedRegimePosterior(
        source_window_id=(available - 1) // 4,
        source_start_cycle=available - 4,
        source_end_cycle=available - 1,
        available_cycle=available,
        probabilities=probabilities,
        model_sha256=MODEL_HASH,
    )


def _candidate(available: int, profile: int = 1, **updates) -> AdaptiveMAPCandidate:
    values = dict(
        method_id="window_map",
        source_window_id=available // 16,
        source_start_cycle=available - 16,
        source_end_cycle=available - 1,
        available_cycle=available,
        image=_image(profile),
        update_macs=1_218,
        private_state_bytes=3_840,
        transient_workspace_bytes=3_072,
        host_update_wallclock_us=900.0,
        window_prequential_score=0.90,
        ewma_prequential_score=1.10,
    )
    values.update(updates)
    return AdaptiveMAPCandidate(**values)


def _step(policy, cycle: int, *, posterior=None, candidate=None, **updates):
    values = dict(
        epoch=cycle,
        syndrome_code=512,
        quadrature_phase_bit=cycle % 2,
        host_heartbeat=True,
    )
    values.update(updates)
    return policy.step(
        RouteACycleInput(
            fast_path=ClosedLoopCycleInput(**values),
            posterior_update=posterior,
            candidate=candidate,
            parameter_update_due=cycle % policy.config.parameter_update_period_cycles
            == 0,
            event_model_sha256=policy.config.event_model_sha256,
        )
    )


def _run(policy, start: int, stop: int, schedule=None):
    schedule = {} if schedule is None else schedule
    rows = []
    for cycle in range(start, stop + 1):
        rows.append(_step(policy, cycle, **schedule.get(cycle, {})))
    return rows


def _open_and_commit_first_candidate(policy):
    schedule = {
        5: {"posterior": _posterior(5)},
        9: {"posterior": _posterior(9)},
        13: {"posterior": _posterior(13)},
        16: {"candidate": _candidate(16)},
        17: {"posterior": _posterior(17)},
    }
    rows = _run(policy, 5, 17, schedule)
    return rows


def test_normal_smooth_hysteresis_stages_real_compiled_image_and_commits() -> None:
    policy = _policy()
    rows = _open_and_commit_first_candidate(policy)
    staged = rows[-2]
    committed = rows[-1]

    assert staged.policy_mode == ADAPTIVE_OPEN
    assert staged.commit_gate_open and staged.candidate_accepted
    assert staged.staged_version == 1
    assert committed.commit_status == "committed"
    assert committed.active_bank_version == 1
    assert committed.readback_status == "confirmed"
    assert committed.deadline.source_to_action_cycles == 6
    assert not committed.deadline.fast_deadline_miss
    assert committed.fast_path_record.map_decision_accepted


def test_tail_promotes_only_continuously_updated_validated_ewma_shadow() -> None:
    policy = _policy()
    _open_and_commit_first_candidate(policy)
    tail = (0.05, 0.05, 0.45, 0.45)
    schedule = {
        21: {"posterior": _posterior(21, tail)},
        25: {"posterior": _posterior(25, tail)},
        29: {"posterior": _posterior(29, tail)},
        32: {
            "candidate": _candidate(
                32,
                2,
                method_id="ewma_adaptive_map",
                window_prequential_score=1.20,
                ewma_prequential_score=0.80,
            )
        },
    }
    rows = _run(policy, 18, 33, schedule)

    first = next(row for row in rows if row.cycle_index == 21)
    staged = next(row for row in rows if row.cycle_index == 32)
    committed = rows[-1]
    assert first.policy_mode == TAIL_TRUSTED and first.fallback_active
    assert first.online_update_frozen and first.trusted_switch_requested
    assert staged.candidate_accepted and staged.candidate_reason == "staged"
    assert staged.staged_version == 2
    assert committed.commit_status == "committed"
    assert committed.trusted_switch_completed
    assert committed.decision_source == "trusted_ewma_shadow"


def test_window_candidate_is_rejected_in_tail_even_on_parameter_boundary() -> None:
    policy = _policy()
    _open_and_commit_first_candidate(policy)
    tail = (0.05, 0.05, 0.45, 0.45)
    schedule = {
        21: {"posterior": _posterior(21, tail)},
        25: {"posterior": _posterior(25, tail)},
        29: {"posterior": _posterior(29, tail)},
        32: {"candidate": _candidate(32, 2)},
    }
    rows = _run(policy, 18, 32, schedule)
    action = rows[-1]
    assert action.candidate_received and not action.candidate_accepted
    assert action.candidate_reason == "tail_or_uncertain_requires_trusted_ewma_shadow"
    assert action.online_update_frozen


def test_leakage_reset_and_ack_do_not_bypass_recovery_hysteresis() -> None:
    policy = _policy()
    _open_and_commit_first_candidate(policy)
    schedule = {
        21: {"posterior": _posterior(21)},
        22: {"syndrome_x": "leakage"},
        23: {"syndrome_x": "leakage"},
        24: {"reset_ack": True},
        25: {"posterior": _posterior(25)},
        29: {"posterior": _posterior(29)},
        32: {"candidate": _candidate(32, 2)},
    }
    rows = _run(policy, 18, 32, schedule)
    leakage_rows = [row for row in rows if row.policy_mode == LEAKAGE_RESET]
    assert leakage_rows
    assert any(row.reset_request for row in leakage_rows)
    ack = next(row for row in rows if row.cycle_index == 24)
    first_recovery = next(row for row in rows if row.cycle_index == 25)
    reopened = next(row for row in rows if row.cycle_index == 29)
    submitted = rows[-1]
    assert ack.policy_mode == RECOVERING and not ack.commit_gate_open
    assert not first_recovery.commit_gate_open
    assert reopened.commit_gate_open
    assert submitted.candidate_accepted


def test_integrity_fault_republishes_lkg_with_monotonic_version() -> None:
    policy = _policy()
    _open_and_commit_first_candidate(policy)
    _run(policy, 18, 22, {21: {"posterior": _posterior(21)}})
    lkg = policy.supervisor.last_known_good_semantics_sha256
    schedule = {
        23: {"reported_integrity_ok": False},
        24: {"reported_integrity_ok": False},
    }
    rows = _run(policy, 23, 25, schedule)
    fault = rows[0]
    staged = rows[1]
    committed = rows[2]
    assert fault.policy_mode == INTEGRITY_ROLLBACK
    assert fault.fallback_active and fault.rollback_requested
    assert staged.safety_stage_accepted and staged.staged_version == 2
    assert committed.active_bank_version == 2
    assert committed.active_semantics_sha256 == lkg
    assert committed.rollback_completed
    assert committed.active_bank_version > fault.active_bank_version


@pytest.mark.parametrize(
    "fault,primary",
    [
        ({"input_crc_ok": False}, "input_crc_mismatch"),
        ({"reported_active_bank_version": 0}, "bank_version_mismatch"),
        ({"reported_parameter_age_cycles": 101}, "parameter_stale"),
        ({"deadline_ok": False}, "deadline_miss"),
    ],
)
def test_integrity_taxonomy_is_fail_closed(fault, primary) -> None:
    policy = _policy()
    _open_and_commit_first_candidate(policy)
    action = _step(policy, 18, **fault)
    assert action.policy_mode == INTEGRITY_ROLLBACK
    assert action.primary_reason == primary
    assert action.fallback_active
    assert not action.commit_gate_open
    assert action.deadline.fast_deadline_miss == (primary == "deadline_miss")


def test_uncertain_posterior_routes_to_trusted_ewma_without_lkg_rollback() -> None:
    policy = _policy()
    _open_and_commit_first_candidate(policy)
    uncertain = (0.30, 0.20, 0.25, 0.25)
    schedule = {
        21: {"posterior": _posterior(21, uncertain)},
        25: {"posterior": _posterior(25, uncertain)},
        29: {"posterior": _posterior(29, uncertain)},
        32: {
            "candidate": _candidate(
                32,
                2,
                method_id="ewma_adaptive_map",
                window_prequential_score=1.10,
                ewma_prequential_score=0.90,
            )
        },
    }
    rows = _run(policy, 18, 33, schedule)
    assert any(row.policy_mode == POSTERIOR_UNCERTAIN for row in rows)
    staged = next(row for row in rows if row.cycle_index == 32)
    assert staged.candidate_accepted and not staged.rollback_requested
    assert rows[-1].decision_source == "trusted_ewma_shadow"
    assert rows[-1].trusted_switch_completed


def test_tail_arriving_before_commit_invalidates_pending_candidate() -> None:
    policy = _policy()
    schedule = {
        5: {"posterior": _posterior(5)},
        9: {"posterior": _posterior(9)},
        13: {"posterior": _posterior(13)},
        16: {"candidate": _candidate(16)},
        17: {"posterior": replace(_posterior(17), source_window_id=99, probabilities=(0.05, 0.05, 0.45, 0.45))},
    }
    rows = _run(policy, 5, 17, schedule)
    assert rows[-2].candidate_accepted
    assert rows[-1].commit_status == "rejected"
    assert rows[-1].commit_reason == "hysteresis_invalidated"
    assert rows[-1].active_bank_version == 0
    assert rows[-1].policy_mode == TAIL_TRUSTED


def test_budget_violation_and_wrong_cadence_fail_closed() -> None:
    policy = _policy()
    _run(
        policy,
        5,
        15,
        {
            5: {"posterior": _posterior(5)},
            9: {"posterior": _posterior(9)},
            13: {"posterior": _posterior(13)},
        },
    )
    oversized = _candidate(16, update_macs=8_193)
    action = _step(policy, 16, candidate=oversized)
    assert not action.candidate_accepted
    assert action.candidate_reason == "candidate_matched_budget_violation"
    assert action.deadline.host_budget_violation

    with pytest.raises(ValueError, match="common cadence"):
        policy.step(
            RouteACycleInput(
                fast_path=ClosedLoopCycleInput(
                    epoch=17, syndrome_code=512, quadrature_phase_bit=1
                ),
                parameter_update_due=True,
            )
        )


@pytest.mark.parametrize(
    "posterior_probabilities,candidate_updates,expected_reason",
    [
        (
            (0.75, 0.15, 0.05, 0.05),
            {},
            "window_router_proof_failed",
        ),
        (
            (0.55, 0.35, 0.05, 0.05),
            {"window_prequential_score": 1.10, "ewma_prequential_score": 0.90},
            "window_router_proof_failed",
        ),
        (
            (0.55, 0.35, 0.05, 0.05),
            {"router_algorithm_sha256": "0" * 64},
            "candidate_router_algorithm_hash_mismatch",
        ),
        (
            (0.55, 0.35, 0.05, 0.05),
            {"update_macs": 1_217},
            "candidate_does_not_account_for_full_dual_shadow_router",
        ),
    ],
)
def test_window_router_proof_and_full_system_budget_fail_closed(
    posterior_probabilities, candidate_updates, expected_reason
) -> None:
    policy = _policy()
    schedule = {
        5: {"posterior": _posterior(5, posterior_probabilities)},
        9: {"posterior": _posterior(9, posterior_probabilities)},
        13: {"posterior": _posterior(13, posterior_probabilities)},
    }
    _run(policy, 5, 15, schedule)
    action = _step(policy, 16, candidate=_candidate(16, **candidate_updates))
    assert action.candidate_received and not action.candidate_accepted
    assert action.candidate_reason == expected_reason
    assert action.active_bank_version == 0


def test_posterior_and_action_schema_reject_noncausal_or_undefined_data() -> None:
    with pytest.raises(ValueError, match="strictly after"):
        ObservedRegimePosterior(1, 0, 3, 3, (0.7, 0.1, 0.1, 0.1), MODEL_HASH)
    policy = _policy()
    bad_sum = ObservedRegimePosterior(1, 1, 4, 5, (0.7, 0.2, 0.1, 0.1), MODEL_HASH)
    with pytest.raises(ValueError, match="sum to one"):
        _step(policy, 5, posterior=bad_sum)


def test_clone_preserves_semantics_but_changes_integrity_identity() -> None:
    original = _image(2, 0)
    clone = clone_parameter_image(original, active_bank_version=9)
    assert parameter_image_semantics_sha256(original) == parameter_image_semantics_sha256(clone)
    assert original.image_sha256 != clone.image_sha256
    assert clone.active_bank_version == 9


def _frozen_policy() -> RegimeAwareSafeAdaptivePolicy:
    return RegimeAwareSafeAdaptivePolicy(
        _image(0),
        config=RouteAPolicyConfig(
            threshold_lock_sha256="1" * 64,
            posterior_model_sha256=MODEL_HASH,
            event_model_sha256="7" * 64,
            threshold_protocol_id="ROUTE-A-TEST-LOCK-V1",
        ),
    )


def test_observed_event_alert_is_upper_policy_tail_and_freezes_updates() -> None:
    policy = _frozen_policy()
    action = _step(policy, 5, ood_score_code=255)
    assert action.primary_reason == "observed_event_score_exceeded"
    assert action.fallback_active
    assert action.online_update_frozen
    assert not action.commit_gate_open
    assert action.decision_source == "initial_static_calibration"
    assert action.trusted_switch_requested


def test_frozen_event_model_hash_mismatch_is_integrity_rollback() -> None:
    policy = _frozen_policy()
    action = policy.step(
        RouteACycleInput(
            fast_path=ClosedLoopCycleInput(
                epoch=5,
                syndrome_code=512,
                quadrature_phase_bit=1,
            ),
            event_model_sha256="0" * 64,
        )
    )
    assert action.primary_reason == "event_model_hash_mismatch"
    assert action.fallback_active
    assert action.rollback_requested
