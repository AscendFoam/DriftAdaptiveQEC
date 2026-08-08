from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from cnn_fpga.hwio import phase9_raw_iq_stream_contract as runtime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9/t9_2_6_raw_iq_platform_envelope.json"
REPORT = ROOT / "docs/t9_2_6_raw_iq_platform_envelope.json"
RELEASE = ROOT / "configs/phase9/t9_2_6_release_pin.json"


def _implementation():
    from cnn_fpga.benchmark import phase9_raw_iq_platform_envelope

    return phase9_raw_iq_platform_envelope


def _report() -> dict:
    if not REPORT.exists():
        pytest.skip("canonical T9.2.6 report has not been generated")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_config_preserves_twin_no_go_and_typed_null_claims() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["task_id"] == "T9.2.6"
    t924 = next(
        row for row in config["parents"] if row["task_id"] == "T9.2.4"
    )
    assert t924["required_verdict"] == "NO_GO_TWIN_QUALIFICATION"
    assert config["arithmetic_contract"]["threshold_values"] is None
    assert all(value is None for value in config["current_claim_state"].values())
    assert config["downstream_release"]["T9.2.7"]["released"] is False
    assert (
        config["downstream_release"]["T-RISK-20260726-01"]["released"]
        is True
    )


def test_rate_profiles_are_two_exact_512ns_profiles() -> None:
    assert set(runtime.RATE_PROFILES) == {
        runtime.RateId.IQ_125_MSPS,
        runtime.RateId.IQ_250_MSPS,
    }
    for profile in runtime.RATE_PROFILES.values():
        assert profile.integration_ns == 512
        assert (
            profile.integration_samples * 1_000_000_000
            // profile.sample_rate_hz
            == 512
        )
        assert (
            profile.sample_rate_hz * profile.axis_cycles_per_sample
            == runtime.AXIS_CLOCK_HZ
        )


@pytest.mark.parametrize(
    ("i_code", "q_code"),
    [
        (-32768, -32768),
        (-32768, 32767),
        (-1, 1),
        (0, 0),
        (32767, -32768),
        (32767, 32767),
    ],
)
def test_iq_pack_is_exact_signed_i16_q16(
    i_code: int, q_code: int
) -> None:
    word = runtime.pack_iq_tdata(i_code, q_code)
    assert 0 <= word < 2**32
    assert runtime.unpack_iq_tdata(word) == (i_code, q_code)


@pytest.mark.parametrize(
    ("i_code", "q_code"),
    [(-32769, 0), (32768, 0), (0, -32769), (0, 32768)],
)
def test_iq_out_of_range_fails_closed(i_code: int, q_code: int) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        runtime.pack_iq_tdata(i_code, q_code)


def test_tuser_layout_is_contiguous_128_bits_and_roundtrips() -> None:
    occupied = set()
    for _name, lsb, bits in runtime.TUSER_FIELDS:
        field = set(range(lsb, lsb + bits))
        assert not occupied & field
        occupied |= field
    assert occupied == set(range(128))
    metadata = runtime.StreamMetadata(
        timestamp=2**48 - 1,
        window_id=2**24 - 1,
        sample_index=255,
        channel_id=15,
        rate_id=1,
        domain_id=2,
        config_version=65535,
        error_flags=65535,
        reset_epoch=255,
    )
    assert runtime.StreamMetadata.unpack(metadata.pack()) == metadata


@pytest.mark.parametrize("profile_id", list(runtime.RateId))
@pytest.mark.parametrize(
    "domain_id",
    [
        runtime.DomainId.SYNTHETIC,
        runtime.DomainId.RECORDED_REPLAY,
        runtime.DomainId.LIVE_RAW,
    ],
)
def test_every_nominal_rate_domain_packet_is_complete(
    profile_id: runtime.RateId, domain_id: runtime.DomainId
) -> None:
    window = runtime.build_window(
        profile_id,
        domain_id=domain_id,
        start_timestamp=100,
        window_id=22,
        config_version=4,
    )
    result = runtime.validate_axis_cycles(
        window, minimum_config_version=4
    )
    assert result.accepted
    assert result.reason is runtime.FaultReason.ACCEPT
    assert result.transfer_count == result.expected_count
    assert window[-1].tlast
    assert all(not row.tlast for row in window[:-1])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("short", runtime.FaultReason.WINDOW_FRAMING_FAILURE),
        ("early_tlast", runtime.FaultReason.WINDOW_FRAMING_FAILURE),
        ("index_gap", runtime.FaultReason.SAMPLE_SEQUENCE_FAILURE),
        ("timestamp_gap", runtime.FaultReason.TIMESTAMP_FAILURE),
        ("mixed_version", runtime.FaultReason.CONFIG_VERSION_FAILURE),
        ("nonzero_channel", runtime.FaultReason.INVALID_METADATA),
        ("invalid_domain", runtime.FaultReason.INVALID_METADATA),
    ],
)
def test_structural_packet_faults_fail_closed(
    mutation: str, reason: runtime.FaultReason
) -> None:
    window = runtime.build_window(runtime.RateId.IQ_125_MSPS)
    if mutation == "short":
        window = window[:-1]
    elif mutation == "early_tlast":
        window[3] = replace(window[3], tlast=True)
    elif mutation == "index_gap":
        meta = runtime.StreamMetadata.unpack(window[4].tuser)
        window[4] = replace(
            window[4], tuser=replace(meta, sample_index=9).pack()
        )
    elif mutation == "timestamp_gap":
        meta = runtime.StreamMetadata.unpack(window[4].tuser)
        window[4] = replace(
            window[4], tuser=replace(meta, timestamp=999).pack()
        )
    elif mutation == "mixed_version":
        meta = runtime.StreamMetadata.unpack(window[4].tuser)
        window[4] = replace(
            window[4], tuser=replace(meta, config_version=2).pack()
        )
    elif mutation == "nonzero_channel":
        meta = runtime.StreamMetadata.unpack(window[0].tuser)
        window[0] = replace(
            window[0], tuser=replace(meta, channel_id=1).pack()
        )
    elif mutation == "invalid_domain":
        window = [
            replace(
                row,
                tuser=replace(
                    runtime.StreamMetadata.unpack(row.tuser),
                    domain_id=int(runtime.DomainId.INVALID_RESERVED),
                ).pack(),
            )
            for row in window
        ]
    result = runtime.validate_axis_cycles(window)
    assert not result.accepted
    assert result.reason is reason


def test_axis_payload_must_remain_stable_while_stalled() -> None:
    window = runtime.build_window(runtime.RateId.IQ_125_MSPS)
    stable = [replace(window[0], tready=False), window[0], *window[1:]]
    assert runtime.validate_axis_cycles(stable).accepted
    unstable = [
        replace(window[0], tready=False),
        window[1],
        *window[2:],
    ]
    result = runtime.validate_axis_cycles(unstable)
    assert result.reason is runtime.FaultReason.AXIS_STABILITY_FAILURE


def test_window_timeout_retires_missing_tlast_fail_closed() -> None:
    window = runtime.build_window(runtime.RateId.IQ_125_MSPS)
    cycles = [
        window[0],
        *[
            runtime.AxisCycle(tvalid=False, tready=True)
            for _ in range(runtime.WINDOW_DEADLINE_AXIS_CYCLES)
        ],
    ]
    result = runtime.validate_axis_cycles(cycles)
    assert result.reason is runtime.FaultReason.WINDOW_TIMEOUT_FAILURE
    candidate = runtime.FastPathObservation(
        syndrome_code=1,
        syndrome_x="e",
        syndrome_z="e",
        quadrature_phase_bit=1,
        ood_score_code=0,
        parameter_age_code=0,
    )
    assert (
        runtime.fail_closed_fast_path_observation(
            candidate, result
        ).observation_valid
        is False
    )


def test_timeout_quarantine_requires_matching_drain_or_exact_reset() -> None:
    first = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=0,
        start_timestamp=0,
        config_version=1,
        reset_epoch=0,
    )
    accepted, state = runtime.validate_and_retire_sequence(
        first, runtime.IngressSequenceState()
    )
    assert accepted.accepted
    second = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=1,
        start_timestamp=200,
        config_version=2,
        reset_epoch=0,
    )
    timeout_cycles = [
        second[0],
        *[
            runtime.AxisCycle(tvalid=False, tready=True)
            for _ in range(runtime.WINDOW_DEADLINE_AXIS_CYCLES)
        ],
    ]
    timeout, quarantined = runtime.validate_and_retire_sequence(
        timeout_cycles, state
    )
    assert timeout.reason is runtime.FaultReason.WINDOW_TIMEOUT_FAILURE
    assert quarantined.quarantined
    assert quarantined.poisoned_window_count == 1

    retry, unchanged = runtime.validate_and_retire_sequence(
        second, quarantined
    )
    assert retry.reason is runtime.FaultReason.QUARANTINE_ACTIVE_FAILURE
    assert unchanged == quarantined

    split_predicate = [
        second[1],
        replace(second[0], tlast=True),
    ]
    split_result, unchanged = runtime.validate_and_retire_sequence(
        split_predicate, unchanged
    )
    assert (
        split_result.reason
        is runtime.FaultReason.QUARANTINE_ACTIVE_FAILURE
    )
    assert unchanged == quarantined

    drain, cleared = runtime.validate_and_retire_sequence(
        [second[-1]], unchanged
    )
    assert drain.reason is runtime.FaultReason.QUARANTINE_ACTIVE_FAILURE
    assert not cleared.quarantined
    assert cleared.poisoned_window_count == 1
    retry_after_drain, retired = runtime.validate_and_retire_sequence(
        second, cleared
    )
    assert retry_after_drain.accepted
    assert retired.poisoned_window_count == 1

    timeout, quarantined = runtime.validate_and_retire_sequence(
        timeout_cycles, state
    )
    partial_reset = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=0,
        start_timestamp=0,
        config_version=1,
        reset_epoch=1,
    )
    incomplete, unchanged = runtime.validate_and_retire_sequence(
        [partial_reset[0]], quarantined
    )
    assert incomplete.reason is runtime.FaultReason.WINDOW_FRAMING_FAILURE
    assert unchanged == quarantined
    bad_config_reset = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=0,
        start_timestamp=0,
        config_version=2,
        reset_epoch=1,
    )
    stale_config, unchanged = runtime.validate_and_retire_sequence(
        bad_config_reset, quarantined
    )
    assert stale_config.reason is runtime.FaultReason.FRESHNESS_REPLAY_FAILURE
    assert unchanged == quarantined
    mixed_epoch_reset = list(partial_reset)
    mixed_metadata = runtime.StreamMetadata.unpack(
        mixed_epoch_reset[20].tuser
    )
    mixed_epoch_reset[20] = replace(
        mixed_epoch_reset[20],
        tuser=replace(mixed_metadata, reset_epoch=0).pack(),
    )
    mixed, unchanged = runtime.validate_and_retire_sequence(
        mixed_epoch_reset, quarantined
    )
    assert mixed.reason is runtime.FaultReason.INVALID_METADATA
    assert unchanged == quarantined
    bad_tlast_reset = list(partial_reset)
    bad_tlast_reset[-1] = replace(bad_tlast_reset[-1], tlast=False)
    malformed, unchanged = runtime.validate_and_retire_sequence(
        bad_tlast_reset, quarantined
    )
    assert malformed.reason is runtime.FaultReason.WINDOW_FRAMING_FAILURE
    assert unchanged == quarantined
    reset = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=0,
        start_timestamp=0,
        config_version=1,
        reset_epoch=1,
    )
    reset_result, reset_state = runtime.validate_and_retire_sequence(
        reset, quarantined
    )
    assert reset_result.accepted
    assert not reset_state.quarantined
    assert reset_state.poisoned_window_count == 1


def test_stateful_sequence_rejects_replay_and_stale_reset_epoch() -> None:
    state = runtime.IngressSequenceState()
    first = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=0,
        start_timestamp=0,
        config_version=1,
        reset_epoch=0,
    )
    result, state = runtime.validate_and_retire_sequence(first, state)
    assert result.accepted
    replay, same_state = runtime.validate_and_retire_sequence(first, state)
    assert replay.reason is runtime.FaultReason.FRESHNESS_REPLAY_FAILURE
    assert same_state == state
    second = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=1,
        start_timestamp=200,
        config_version=2,
        reset_epoch=0,
    )
    result, state = runtime.validate_and_retire_sequence(second, state)
    assert result.accepted
    reset = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        window_id=0,
        start_timestamp=0,
        config_version=2,
        reset_epoch=1,
    )
    result, state = runtime.validate_and_retire_sequence(reset, state)
    assert result.accepted
    stale, unchanged = runtime.validate_and_retire_sequence(second, state)
    assert stale.reason is runtime.FaultReason.FRESHNESS_REPLAY_FAILURE
    assert unchanged == state


def test_mixed_reset_epoch_old_fifo_beat_is_rejected() -> None:
    window = runtime.build_window(
        runtime.RateId.IQ_125_MSPS, reset_epoch=1
    )
    metadata = runtime.StreamMetadata.unpack(window[20].tuser)
    window[20] = replace(
        window[20], tuser=replace(metadata, reset_epoch=0).pack()
    )
    result = runtime.validate_axis_cycles(window)
    assert result.reason is runtime.FaultReason.INVALID_METADATA


@pytest.mark.parametrize("flag_name", sorted(runtime.ERROR_FLAG_BITS))
def test_every_explicit_error_flag_poison_whole_window(
    flag_name: str,
) -> None:
    window = runtime.build_window(runtime.RateId.IQ_125_MSPS)
    metadata = runtime.StreamMetadata.unpack(window[-1].tuser)
    window[-1] = replace(
        window[-1],
        tuser=replace(
            metadata,
            error_flags=1 << runtime.ERROR_FLAG_BITS[flag_name],
        ).pack(),
    )
    result = runtime.validate_axis_cycles(window)
    assert not result.accepted
    assert result.reason is not runtime.FaultReason.ACCEPT
    candidate = runtime.FastPathObservation(
        syndrome_code=1023,
        syndrome_x="leakage",
        syndrome_z="leakage",
        quadrature_phase_bit=1,
        ood_score_code=0,
        parameter_age_code=7,
    )
    closed = runtime.fail_closed_fast_path_observation(candidate, result)
    from cnn_fpga.runtime.bit_accurate_hardware_reference import (
        decode_input_word,
    )

    decoded = decode_input_word(closed.pack_legacy_58bit_word())
    assert decoded.input_crc_ok
    assert decoded.observation_valid is False
    assert decoded.deadline_ok is False


def test_error_priority_is_not_order_of_detection() -> None:
    window = runtime.build_window(runtime.RateId.IQ_125_MSPS)
    metadata = runtime.StreamMetadata.unpack(window[0].tuser)
    flags = (
        1 << runtime.ERROR_FLAG_BITS["reset_mid_window"]
        | 1 << runtime.ERROR_FLAG_BITS["cdc_overflow"]
        | 1 << runtime.ERROR_FLAG_BITS["tlast_missing"]
    )
    window[0] = replace(
        window[0],
        tuser=replace(metadata, error_flags=flags).pack(),
    )
    assert (
        runtime.validate_axis_cycles(window).reason
        is runtime.FaultReason.RESET_MID_WINDOW
    )
    assert (
        runtime.FAULT_PRIORITY[0]
        is runtime.FaultReason.AXIS_STABILITY_FAILURE
    )
    assert (
        runtime.FAULT_PRIORITY[1]
        is runtime.FaultReason.WINDOW_TIMEOUT_FAILURE
    )
    assert (
        runtime.FAULT_PRIORITY[2]
        is runtime.FaultReason.QUARANTINE_ACTIVE_FAILURE
    )
    assert runtime.FAULT_PRIORITY[3] is runtime.FaultReason.RESET_MID_WINDOW
    assert runtime.FAULT_PRIORITY[-1] is runtime.FaultReason.ACCEPT


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (8, 0),
        (9, 1),
        (23, 1),
        (24, 2),
        (-8, 0),
        (-9, -1),
        (-23, -1),
        (-24, -2),
        (10**9, 127),
        (-(10**9), -128),
    ],
)
def test_ties_even_and_saturation(value: int, expected: int) -> None:
    assert (
        runtime.round_shift_ties_to_even_saturate(
            value, shift=4, output_bits=8
        )
        == expected
    )


def test_complex_matched_filter_conjugate_sign_and_extreme_accumulation() -> None:
    profile = runtime.RATE_PROFILES[runtime.RateId.IQ_250_MSPS]
    samples = [(32767, -32768)] * profile.integration_samples
    h_i = [131071] * profile.integration_samples
    h_q = [-131072] * profile.integration_samples
    result = runtime.matched_filter_accumulate(
        runtime.RateId.IQ_250_MSPS, samples, h_i, h_q
    )
    expected_i = profile.integration_samples * (
        32767 * 131071 + (-32768) * (-131072)
    )
    expected_q = profile.integration_samples * (
        (-32768) * 131071 - 32767 * (-131072)
    )
    assert result.accumulator_i_q16_32 == expected_i
    assert result.accumulator_q_q16_32 == expected_q
    assert result.sticky_overflow is False
    assert -(1 << 47) <= expected_i < (1 << 47)
    assert -(1 << 47) <= expected_q < (1 << 47)


def test_calibration_66x67bit_path_identity_and_saturation() -> None:
    identity = runtime.calibrate_accumulators_to_q8_16(
        3 << 32,
        -(2 << 32),
        (65536, 0, 0, 65536),
        (0, 0),
    )
    assert (identity.i_q8_16, identity.q_q8_16) == (
        3 << 16,
        -(2 << 16),
    )
    assert identity.sticky_overflow is False
    saturated = runtime.calibrate_accumulators_to_q8_16(
        (1 << 47) - 1,
        (1 << 47) - 1,
        ((1 << 17) - 1,) * 4,
        ((1 << 23) - 1,) * 2,
    )
    assert saturated.i_q8_16 == (1 << 23) - 1
    assert saturated.q_q8_16 == (1 << 23) - 1
    assert saturated.sticky_overflow is True


def _package(
    profile_id: runtime.RateId, *, qualified: bool = False
) -> runtime.FrontendBankPackage:
    count = runtime.RATE_PROFILES[profile_id].integration_samples
    return runtime.FrontendBankPackage(
        schema_version=1,
        config_version=1,
        activation_window=2,
        profile_id=int(profile_id),
        coefficient_i_q1_17=(131071,) + (0,) * (count - 1),
        coefficient_q_q1_17=(0,) * count,
        calibration_matrix_q2_16=(65536, 0, 0, 65536),
        calibration_offset_q8_16=(0, 0),
        discriminator_thresholds_q8_16=(0, 0, 0, 0),
        threshold_qualification_state=1 if qualified else 0,
        qualification_receipt_sha256="1" * 64 if qualified else None,
    )


@pytest.mark.parametrize("profile_id", list(runtime.RateId))
def test_complete_package_crc_is_stable_and_bit_sensitive(
    profile_id: runtime.RateId,
) -> None:
    package = _package(profile_id)
    assert package.payload_bytes()
    assert package.crc32() == package.crc32()
    mutated = replace(package, config_version=2)
    assert mutated.crc32() != package.crc32()
    with pytest.raises(ValueError, match="thresholds must remain exact zero"):
        replace(
            package,
            discriminator_thresholds_q8_16=(1, 0, 0, 0),
        ).payload_bytes()
    with pytest.raises(ValueError, match="coefficient energy"):
        replace(
            package,
            coefficient_i_q1_17=(0,)
            * runtime.RATE_PROFILES[profile_id].integration_samples,
        ).payload_bytes()


def test_atomic_bank_commit_and_lkg_rollback() -> None:
    state = runtime.BankCommitState()
    package = _package(runtime.RateId.IQ_125_MSPS, qualified=True)
    assert runtime.TRUSTED_QUALIFICATION_RECEIPT_SHA256 is None
    with pytest.raises(ValueError, match="sealed trusted hash"):
        state.commit(
            requested_bank=1,
            package=package,
            presented_crc32=package.crc32(),
            next_window_id=2,
        )
    preloaded = runtime.BankCommitState(
        active_bank=1,
        active_version=1,
        lkg_bank=0,
        lkg_version=0,
        active_package_sha256=package.sha256(),
        active_profile_id=int(runtime.RateId.IQ_125_MSPS),
    )
    assert preloaded.rollback_lkg() == runtime.BankCommitState()
    with pytest.raises(ValueError, match="between windows"):
        replace(state, window_open=True).commit(
            requested_bank=1,
            package=package,
            presented_crc32=package.crc32(),
            next_window_id=2,
        )
    with pytest.raises(ValueError, match="does not match"):
        state.commit(
            requested_bank=1,
            package=package,
            presented_crc32=package.crc32() ^ 1,
            next_window_id=2,
        )
    with pytest.raises(ValueError, match="exactly one"):
        wrong_version = replace(package, config_version=2)
        state.commit(
            requested_bank=1,
            package=wrong_version,
            presented_crc32=wrong_version.crc32(),
            next_window_id=2,
        )
    with pytest.raises(ValueError, match="activation_window"):
        state.commit(
            requested_bank=1,
            package=package,
            presented_crc32=package.crc32(),
            next_window_id=3,
        )
    with pytest.raises(ValueError, match="inactive"):
        state.commit(
            requested_bank=0,
            package=package,
            presented_crc32=package.crc32(),
            next_window_id=2,
        )
    with pytest.raises(ValueError, match="inactive"):
        state.commit(
            requested_bank=True,
            package=package,
            presented_crc32=package.crc32(),
            next_window_id=2,
        )
    blocked = _package(runtime.RateId.IQ_125_MSPS)
    with pytest.raises(ValueError, match="unqualified"):
        state.commit(
            requested_bank=1,
            package=blocked,
            presented_crc32=blocked.crc32(),
            next_window_id=2,
        )
    with pytest.raises(ValueError, match="wrap"):
        runtime.BankCommitState(
            active_bank=0,
            active_version=0xFFFF,
            lkg_bank=1,
            lkg_version=0xFFFE,
        ).commit(
            requested_bank=1,
            package=package,
            presented_crc32=package.crc32(),
            next_window_id=2,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "schema_version",
        "config_version",
        "activation_window",
        "profile_id",
        "threshold_qualification_state",
    ],
)
def test_package_integer_fields_reject_bool_alias(field_name: str) -> None:
    package = _package(runtime.RateId.IQ_125_MSPS, qualified=True)
    with pytest.raises((TypeError, ValueError)):
        replace(package, **{field_name: True}).payload_bytes()


@pytest.mark.parametrize(
    "field_name",
    [
        "active_bank",
        "active_version",
        "lkg_bank",
        "lkg_version",
        "active_profile_id",
        "lkg_profile_id",
    ],
)
def test_bank_state_integer_fields_reject_bool_alias(
    field_name: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        runtime.BankCommitState(**{field_name: True})


@pytest.mark.parametrize("field_name", ["tdata", "tuser"])
def test_axis_integer_fields_reject_bool_alias(field_name: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        runtime.AxisCycle(**{field_name: True})


def test_rate_and_domain_identifiers_reject_bool_alias() -> None:
    with pytest.raises(TypeError, match="profile_id"):
        runtime.build_window(True)
    with pytest.raises(TypeError, match="domain_id"):
        runtime.build_window(runtime.RateId.IQ_125_MSPS, domain_id=True)
    with pytest.raises(TypeError, match="profile_id"):
        runtime.matched_filter_accumulate(True, [], [], [])


def test_legacy_adapter_is_exact_58bit_crc_schema() -> None:
    schema = runtime.legacy_fast_path_layout()
    assert schema["word_bits"] == 58
    assert schema["payload_bits"] == 42
    observation = runtime.FastPathObservation(
        syndrome_code=1023,
        syndrome_x="leakage",
        syndrome_z="e",
        quadrature_phase_bit=1,
        ood_score_code=255,
        parameter_age_code=65535,
        reset_ack=True,
        observation_valid=True,
        deadline_ok=True,
    )
    word = observation.pack_legacy_58bit_word()
    assert word.bit_length() <= 58
    assert runtime.verify_fast_path_roundtrip(observation)
    from cnn_fpga.runtime.bit_accurate_hardware_reference import (
        decode_input_word,
    )

    assert not decode_input_word(word ^ (1 << 42)).input_crc_ok


def test_latency_boundaries_do_not_transfer_six_cycles_to_raw_iq() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = {
        row["boundary_id"]: row for row in config["latency_boundaries"]
    }
    assert rows["FAST_PATH_CORE_INPUT_TO_ACTION"]["cycles"] == 6
    assert rows["DISCRIMINATOR_OUT_TO_ACTION"]["cycles"] == 6
    for name in (
        "ADC_LAST_SAMPLE_TO_TRIGGER",
        "RAW_IQ_SOURCE_TO_TRIGGER",
    ):
        assert rows[name]["cycles"] is None
        assert rows[name]["measured_ns"] is None


def test_platform_candidates_are_not_results_and_gw2ar_is_excluded() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    intersection = config["platform_intersection"]
    assert {
        row["platform_id"] for row in intersection["candidate_classes"]
    } == {"AMD_ZCU111_XCZU28DR", "AMD_ZCU216_XCZU49DR"}
    assert all(
        row["qualification_status"]
        == "SOURCE_CONFIRMED_NOT_SELECTED_NOT_BUILT"
        for row in intersection["candidate_classes"]
    )
    assert intersection["excluded_platforms"][0]["platform_id"] == (
        "Tang_Nano_20K_GW2AR"
    )
    assert intersection["budget_status"].startswith("DESIGN_ENVELOPE_")


def test_canonical_report_release_and_all_gates_verify() -> None:
    implementation = _implementation()
    report = _report()
    assert RELEASE.exists()
    checks = implementation.verify_report(
        REPORT, expected_analysis_sha256=report["analysis_sha256"]
    )
    assert tuple(checks) == implementation.GATE_IDS
    assert all(checks.values())


def test_report_contains_full_enumeration_and_adversarial_receipts() -> None:
    audit = _report()["executable_audit"]
    assert audit["iq_signed_code_roundtrips"] == 131072
    assert audit["tuser_boundary_roundtrips"] == 18
    assert audit["rounding_exhaustive_cases"] == 262144
    assert len(audit["nominal_rate_domain_cases"]) == 6
    assert len(audit["fault_flag_cases"]) == 16
    assert len(audit["structural_failure_cases"]) == 8
    assert audit["stateful_freshness_cases"] == 5
    assert audit["mixed_reset_epoch_rejected"] is True
    assert audit["window_timeout_rejected"] is True
    assert audit["matched_filter_reference_cases"] == 516
    assert audit["calibration_reference_cases"] == 729
    assert audit["unsafe_commit_cases_rejected"] == 9
    assert audit["successful_commit_count"] == 0
    assert audit["trusted_qualification_receipt_sha256"] is None
    assert audit["timeout_quarantine_cases"] == 10
    assert audit["timeout_poison_count"] == 1
    assert audit["strict_bool_alias_cases_rejected"] == 16
    assert audit["all_passed"] is True


def test_mutation_replay_is_one_to_one_complete_and_targeted() -> None:
    implementation = _implementation()
    audit = _report()["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(
        implementation.GATE_IDS
    )
    assert audit["all_detected"] is True
    assert [row["target_gate"] for row in audit["records"]] == list(
        implementation.GATE_IDS
    )
    assert all(
        row["target_gate"] in row["failed_gates"]
        for row in audit["records"]
    )


def test_semantic_tampering_is_rejected_without_rehash_rescue() -> None:
    implementation = _implementation()
    candidate = copy.deepcopy(_report())
    candidate["claim_state"]["external_sota"] = False
    candidate["analysis_sha256"] = implementation._analysis_sha(candidate)
    with pytest.raises(TypeError, match="canonical report path"):
        implementation.verify_report(candidate)
    checks = implementation._check_gates(
        candidate, verify_live=False, verify_outputs=False
    )
    assert checks[
        "G32_all_performance_physical_hardware_puviani_sota_rank_fields_are_null"
    ] is False


def test_source_data_reconstructs_snapshot_losslessly() -> None:
    implementation = _implementation()
    report = _report()
    reconstructed = implementation._read_source_data(
        ROOT / report["source_data"]["path"]
    )
    assert reconstructed == implementation._snapshot(report)
