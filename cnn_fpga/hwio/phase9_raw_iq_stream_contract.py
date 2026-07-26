"""Executable T9.2.6 raw-IQ/AXI-stream interface contract.

This module deliberately stops at the discriminator-to-fast-path adapter.  It
does not implement or tune a matched filter, infer thresholds from the failed
T9.2.4 twin, or claim board timing.  Its purpose is to make the frozen packet,
fixed-point, arithmetic, versioning, and fail-closed semantics executable
before the synthesizable T9.2.7 frontend is written.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import zlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from cnn_fpga.runtime.bit_accurate_hardware_reference import (
    INPUT_SCHEMA,
    decode_input_word,
    encode_input_word,
)


AXIS_CLOCK_HZ = 250_000_000
AXIS_TDATA_BITS = 32
AXIS_TUSER_BITS = 128
TIMESTAMP_TICK_NS = 4
MAX_WINDOW_SAMPLES = 128
MIN_ELASTIC_BUFFER_WINDOWS = 2
MIN_ELASTIC_BUFFER_BEATS = MAX_WINDOW_SAMPLES * MIN_ELASTIC_BUFFER_WINDOWS
WINDOW_DEADLINE_AXIS_CYCLES = 192
# Deliberately null after the T9.2.4 scientific NO-GO.  A fresh qualification
# must amend this exact constant before any frontend bank can activate.
TRUSTED_QUALIFICATION_RECEIPT_SHA256: str | None = None


class RateId(enum.IntEnum):
    """Only the two frozen 512 ns integration profiles are legal."""

    IQ_125_MSPS = 0
    IQ_250_MSPS = 1


@dataclasses.dataclass(frozen=True)
class RateProfile:
    rate_id: RateId
    sample_rate_hz: int
    integration_samples: int
    integration_ns: int
    axis_cycles_per_sample: int


RATE_PROFILES: dict[RateId, RateProfile] = {
    RateId.IQ_125_MSPS: RateProfile(
        rate_id=RateId.IQ_125_MSPS,
        sample_rate_hz=125_000_000,
        integration_samples=64,
        integration_ns=512,
        axis_cycles_per_sample=2,
    ),
    RateId.IQ_250_MSPS: RateProfile(
        rate_id=RateId.IQ_250_MSPS,
        sample_rate_hz=250_000_000,
        integration_samples=128,
        integration_ns=512,
        axis_cycles_per_sample=1,
    ),
}


class DomainId(enum.IntEnum):
    SYNTHETIC = 0
    RECORDED_REPLAY = 1
    LIVE_RAW = 2
    INVALID_RESERVED = 3


TUSER_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("timestamp", 0, 48),
    ("window_id", 48, 24),
    ("sample_index", 72, 8),
    ("channel_id", 80, 4),
    ("rate_id", 84, 2),
    ("domain_id", 86, 2),
    ("config_version", 88, 16),
    ("error_flags", 104, 16),
    ("reset_epoch", 120, 8),
)


ERROR_FLAG_BITS: dict[str, int] = {
    "adc_overrange": 0,
    "rfdc_overflow": 1,
    "source_clock_unlock": 2,
    "tlast_early": 3,
    "tlast_missing": 4,
    "length_mismatch": 5,
    "index_gap": 6,
    "index_duplicate": 7,
    "index_reorder": 8,
    "timestamp_regression": 9,
    "config_version_stale": 10,
    "coefficient_crc_failure": 11,
    "calibration_crc_failure": 12,
    "cdc_overflow": 13,
    "reset_mid_window": 14,
    "transport_poison": 15,
}


class FaultReason(enum.IntEnum):
    ACCEPT = 0
    RESET_MID_WINDOW = 1
    CDC_OVERFLOW = 2
    SOURCE_OR_RFDC_FAILURE = 3
    PACKAGE_INTEGRITY_FAILURE = 4
    CONFIG_VERSION_FAILURE = 5
    TIMESTAMP_FAILURE = 6
    SAMPLE_SEQUENCE_FAILURE = 7
    WINDOW_FRAMING_FAILURE = 8
    INVALID_METADATA = 9
    INPUT_QUALITY_FAILURE = 10
    AXIS_STABILITY_FAILURE = 11
    FRESHNESS_REPLAY_FAILURE = 12
    WINDOW_TIMEOUT_FAILURE = 13
    QUARANTINE_ACTIVE_FAILURE = 14


FAULT_PRIORITY: tuple[FaultReason, ...] = (
    FaultReason.AXIS_STABILITY_FAILURE,
    FaultReason.WINDOW_TIMEOUT_FAILURE,
    FaultReason.QUARANTINE_ACTIVE_FAILURE,
    FaultReason.RESET_MID_WINDOW,
    FaultReason.CDC_OVERFLOW,
    FaultReason.SOURCE_OR_RFDC_FAILURE,
    FaultReason.PACKAGE_INTEGRITY_FAILURE,
    FaultReason.CONFIG_VERSION_FAILURE,
    FaultReason.INVALID_METADATA,
    FaultReason.FRESHNESS_REPLAY_FAILURE,
    FaultReason.TIMESTAMP_FAILURE,
    FaultReason.SAMPLE_SEQUENCE_FAILURE,
    FaultReason.WINDOW_FRAMING_FAILURE,
    FaultReason.INPUT_QUALITY_FAILURE,
    FaultReason.ACCEPT,
)


@dataclasses.dataclass(frozen=True)
class StreamMetadata:
    timestamp: int
    window_id: int
    sample_index: int
    channel_id: int
    rate_id: int
    domain_id: int
    config_version: int
    error_flags: int = 0
    reset_epoch: int = 0

    def pack(self) -> int:
        values = dataclasses.asdict(self)
        word = 0
        for name, lsb, bits in TUSER_FIELDS:
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < (1 << bits)
            ):
                raise ValueError(f"{name} does not fit unsigned {bits} bits")
            word |= value << lsb
        return word

    @classmethod
    def unpack(cls, word: int) -> "StreamMetadata":
        if isinstance(word, bool) or not isinstance(word, int):
            raise TypeError("TUSER must be an integer")
        if not 0 <= word < (1 << AXIS_TUSER_BITS):
            raise ValueError("TUSER must fit 128 bits")
        values = {
            name: (word >> lsb) & ((1 << bits) - 1)
            for name, lsb, bits in TUSER_FIELDS
        }
        return cls(**values)


def _signed_to_twos(value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("fixed-point code must be an integer")
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if not minimum <= value <= maximum:
        raise ValueError(f"value {value} does not fit signed {bits} bits")
    return value & ((1 << bits) - 1)


def _twos_to_signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def pack_iq_tdata(i_q1_15: int, q_q1_15: int) -> int:
    """Pack I in bits 15:0 and Q in bits 31:16."""

    return _signed_to_twos(i_q1_15, 16) | (
        _signed_to_twos(q_q1_15, 16) << 16
    )


def unpack_iq_tdata(word: int) -> tuple[int, int]:
    if isinstance(word, bool) or not isinstance(word, int):
        raise TypeError("TDATA must be an integer")
    if not 0 <= word < (1 << AXIS_TDATA_BITS):
        raise ValueError("TDATA must fit 32 bits")
    return _twos_to_signed(word, 16), _twos_to_signed(word >> 16, 16)


@dataclasses.dataclass(frozen=True)
class AxisCycle:
    """One cycle of the project boundary, including handshake state."""

    tdata: int = 0
    tuser: int = 0
    tlast: bool = False
    tvalid: bool = False
    tready: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.tdata, bool)
            or not isinstance(self.tdata, int)
            or not 0 <= self.tdata < (1 << AXIS_TDATA_BITS)
        ):
            raise ValueError("tdata must fit 32 bits")
        if (
            isinstance(self.tuser, bool)
            or not isinstance(self.tuser, int)
            or not 0 <= self.tuser < (1 << AXIS_TUSER_BITS)
        ):
            raise ValueError("tuser must fit 128 bits")
        for name in ("tlast", "tvalid", "tready"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

    @property
    def transferred(self) -> bool:
        return self.tvalid and self.tready

    @property
    def payload_tuple(self) -> tuple[int, int, bool]:
        return self.tdata, self.tuser, self.tlast


@dataclasses.dataclass(frozen=True)
class WindowValidation:
    accepted: bool
    reason: FaultReason
    transfer_count: int
    expected_count: int | None
    rate_id: int | None
    domain_id: int | None
    window_id: int | None
    config_version: int | None
    aggregated_error_flags: int
    detail: str


def _result(
    reason: FaultReason,
    transfers: Sequence[AxisCycle],
    *,
    expected_count: int | None = None,
    metadata: StreamMetadata | None = None,
    error_flags: int = 0,
    detail: str,
) -> WindowValidation:
    return WindowValidation(
        accepted=reason is FaultReason.ACCEPT,
        reason=reason,
        transfer_count=len(transfers),
        expected_count=expected_count,
        rate_id=None if metadata is None else metadata.rate_id,
        domain_id=None if metadata is None else metadata.domain_id,
        window_id=None if metadata is None else metadata.window_id,
        config_version=None if metadata is None else metadata.config_version,
        aggregated_error_flags=error_flags,
        detail=detail,
    )


def _flag(mask: int, name: str) -> bool:
    return bool(mask & (1 << ERROR_FLAG_BITS[name]))


def validate_transferred_window(
    transfers: Sequence[AxisCycle],
    *,
    minimum_config_version: int = 0,
) -> WindowValidation:
    """Validate one complete, already-handshaken window.

    The first failed category follows ``FAULT_PRIORITY``.  No prefix is
    accepted as a shorter window and no error-bearing window is postselected.
    """

    if not transfers:
        return _result(
            FaultReason.WINDOW_FRAMING_FAILURE,
            transfers,
            detail="empty transfer sequence",
        )
    if any(not cycle.transferred for cycle in transfers):
        return _result(
            FaultReason.AXIS_STABILITY_FAILURE,
            transfers,
            detail="non-transferred cycle passed as a window beat",
        )
    metadata = [StreamMetadata.unpack(cycle.tuser) for cycle in transfers]
    first = metadata[0]
    flags = 0
    for row in metadata:
        flags |= row.error_flags

    # Exact fail-closed priority: high-level transport/reset/integrity faults
    # are decided before structural framing diagnostics.
    if _flag(flags, "reset_mid_window"):
        return _result(
            FaultReason.RESET_MID_WINDOW,
            transfers,
            metadata=first,
            error_flags=flags,
            detail="reset was asserted before TLAST retirement",
        )
    if _flag(flags, "cdc_overflow"):
        return _result(
            FaultReason.CDC_OVERFLOW,
            transfers,
            metadata=first,
            error_flags=flags,
            detail="whole window is poisoned after CDC overflow",
        )
    if _flag(flags, "rfdc_overflow") or _flag(flags, "source_clock_unlock"):
        return _result(
            FaultReason.SOURCE_OR_RFDC_FAILURE,
            transfers,
            metadata=first,
            error_flags=flags,
            detail="RFDC overflow or source clock unlock",
        )
    if _flag(flags, "coefficient_crc_failure") or _flag(
        flags, "calibration_crc_failure"
    ):
        return _result(
            FaultReason.PACKAGE_INTEGRITY_FAILURE,
            transfers,
            metadata=first,
            error_flags=flags,
            detail="coefficient/calibration package failed CRC",
        )
    if (
        _flag(flags, "config_version_stale")
        or first.config_version < minimum_config_version
        or any(row.config_version != first.config_version for row in metadata)
    ):
        return _result(
            FaultReason.CONFIG_VERSION_FAILURE,
            transfers,
            metadata=first,
            error_flags=flags,
            detail="stale or mixed config version",
        )

    try:
        rate = RATE_PROFILES[RateId(first.rate_id)]
        domain = DomainId(first.domain_id)
    except ValueError:
        rate = None
        domain = DomainId.INVALID_RESERVED
    if (
        rate is None
        or domain is DomainId.INVALID_RESERVED
        or first.channel_id != 0
        or any(row.channel_id != first.channel_id for row in metadata)
        or any(row.window_id != first.window_id for row in metadata)
        or any(row.rate_id != first.rate_id for row in metadata)
        or any(row.domain_id != first.domain_id for row in metadata)
        or any(row.reset_epoch != first.reset_epoch for row in metadata)
    ):
        return _result(
            FaultReason.INVALID_METADATA,
            transfers,
            metadata=first,
            error_flags=flags,
            detail="reserved/unknown or mixed metadata",
        )

    assert rate is not None
    expected = rate.integration_samples
    timestamps = [row.timestamp for row in metadata]
    expected_timestamps = [
        first.timestamp + index * rate.axis_cycles_per_sample
        for index in range(len(metadata))
    ]
    if (
        _flag(flags, "timestamp_regression")
        or timestamps != expected_timestamps
        or any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:]))
    ):
        return _result(
            FaultReason.TIMESTAMP_FAILURE,
            transfers,
            expected_count=expected,
            metadata=first,
            error_flags=flags,
            detail="timestamp is not contiguous in 250 MHz clock ticks",
        )

    indices = [row.sample_index for row in metadata]
    if (
        any(
            _flag(flags, name)
            for name in ("index_gap", "index_duplicate", "index_reorder")
        )
        or indices != list(range(len(metadata)))
    ):
        return _result(
            FaultReason.SAMPLE_SEQUENCE_FAILURE,
            transfers,
            expected_count=expected,
            metadata=first,
            error_flags=flags,
            detail="sample indices are not exactly 0..N-1",
        )

    tlast_positions = [
        index for index, cycle in enumerate(transfers) if cycle.tlast
    ]
    if (
        any(
            _flag(flags, name)
            for name in ("tlast_early", "tlast_missing", "length_mismatch")
        )
        or len(transfers) != expected
        or tlast_positions != [expected - 1]
    ):
        return _result(
            FaultReason.WINDOW_FRAMING_FAILURE,
            transfers,
            expected_count=expected,
            metadata=first,
            error_flags=flags,
            detail="length/TLAST is not the exact frozen window",
        )

    if _flag(flags, "adc_overrange") or _flag(flags, "transport_poison"):
        return _result(
            FaultReason.INPUT_QUALITY_FAILURE,
            transfers,
            expected_count=expected,
            metadata=first,
            error_flags=flags,
            detail="ADC overrange or upstream poison",
        )
    return _result(
        FaultReason.ACCEPT,
        transfers,
        expected_count=expected,
        metadata=first,
        error_flags=flags,
        detail="complete error-free window",
    )


def validate_axis_cycles(
    cycles: Sequence[AxisCycle],
    *,
    minimum_config_version: int = 0,
) -> WindowValidation:
    """Validate AXI stability and then the complete transferred packet."""

    pending: tuple[int, int, bool] | None = None
    transfers: list[AxisCycle] = []
    first_valid_cycle: int | None = None
    for cycle_index, cycle in enumerate(cycles):
        if cycle.tvalid and first_valid_cycle is None:
            first_valid_cycle = cycle_index
        if (
            first_valid_cycle is not None
            and cycle_index - first_valid_cycle
            >= WINDOW_DEADLINE_AXIS_CYCLES
            and not any(row.tlast for row in transfers)
        ):
            return _result(
                FaultReason.WINDOW_TIMEOUT_FAILURE,
                transfers,
                detail="window did not retire TLAST before the frozen deadline",
            )
        if pending is not None:
            if not cycle.tvalid or cycle.payload_tuple != pending:
                return _result(
                    FaultReason.AXIS_STABILITY_FAILURE,
                    transfers,
                    detail="TVALID/payload changed while TREADY was low",
                )
        if cycle.tvalid and not cycle.tready:
            pending = cycle.payload_tuple
        elif cycle.transferred:
            transfers.append(cycle)
            pending = None
        else:
            pending = None
    if pending is not None:
        return _result(
            FaultReason.AXIS_STABILITY_FAILURE,
            transfers,
            detail="stream ended with an unaccepted valid beat",
        )
    return validate_transferred_window(
        transfers, minimum_config_version=minimum_config_version
    )


@dataclasses.dataclass(frozen=True)
class IngressSequenceState:
    """Stateful freshness receipt across complete frontend windows."""

    initialized: bool = False
    reset_epoch: int = 0
    last_window_id: int = 0
    last_final_timestamp: int = 0
    config_version: int = 0
    quarantined: bool = False
    quarantined_reset_epoch: int | None = None
    quarantined_window_id: int | None = None
    poisoned_window_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.initialized, bool) or not isinstance(
            self.quarantined, bool
        ):
            raise TypeError("initialized/quarantined must be bool")
        for name, bits in (
            ("reset_epoch", 8),
            ("last_window_id", 24),
            ("last_final_timestamp", 48),
            ("config_version", 16),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be integer")
            if not 0 <= value < (1 << bits):
                raise ValueError(f"{name} must fit {bits} bits")
        for name, bits in (
            ("quarantined_reset_epoch", 8),
            ("quarantined_window_id", 24),
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < (1 << bits)
            ):
                raise ValueError(f"{name} must fit {bits} bits or be null")
        if self.quarantined != (
            self.quarantined_reset_epoch is not None
            and self.quarantined_window_id is not None
        ):
            raise ValueError("quarantine flag and receipt fields must agree")
        if (
            isinstance(self.poisoned_window_count, bool)
            or not isinstance(self.poisoned_window_count, int)
            or self.poisoned_window_count < 0
        ):
            raise ValueError("poisoned_window_count must be nonnegative integer")


def validate_and_retire_sequence(
    cycles: Sequence[AxisCycle],
    state: IngressSequenceState,
    *,
    minimum_config_version: int = 0,
) -> tuple[WindowValidation, IngressSequenceState]:
    """Validate one window plus replay/reset/config freshness.

    Window and timestamp wrap are not implicit.  A reset epoch must increment
    by exactly one, flush the FIFO, and restart at ``window_id=0``; the 8-bit
    epoch itself may not wrap inside one run.
    """

    transferred = [row for row in cycles if row.transferred]
    atomic_reset_origin: IngressSequenceState | None = None
    if state.quarantined:
        metadata = (
            [StreamMetadata.unpack(row.tuser) for row in transferred]
            if transferred
            else []
        )
        drained = bool(
            metadata
            and all(
                meta.reset_epoch == state.quarantined_reset_epoch
                and meta.window_id == state.quarantined_window_id
                and meta.sample_index != 0
                for meta in metadata
            )
            and any(
                row.tlast
                and meta.reset_epoch == state.quarantined_reset_epoch
                and meta.window_id == state.quarantined_window_id
                and meta.sample_index != 0
                for row, meta in zip(transferred, metadata)
            )
        )
        reset_flush = bool(
            metadata
            and state.quarantined_reset_epoch is not None
            and state.quarantined_reset_epoch < 0xFF
            and metadata[0].reset_epoch
            == state.quarantined_reset_epoch + 1
            and metadata[0].window_id == 0
        )
        if not drained and not reset_flush:
            first = metadata[0] if metadata else None
            return (
                _result(
                    FaultReason.QUARANTINE_ACTIVE_FAILURE,
                    transferred,
                    metadata=first,
                    detail=(
                        "timed-out window remains quarantined until its TLAST "
                        "or an exact next reset-epoch flush"
                    ),
                ),
                state,
            )
        cleared = dataclasses.replace(
            state,
            quarantined=False,
            quarantined_reset_epoch=None,
            quarantined_window_id=None,
        )
        if drained:
            first = metadata[0] if metadata else None
            return (
                _result(
                    FaultReason.QUARANTINE_ACTIVE_FAILURE,
                    transferred,
                    metadata=first,
                    detail="affected TLAST drained; no candidate action executed",
                ),
                cleared,
            )
        # Reset-based release is transactional: the candidate must pass the
        # entire packet and cross-window freshness checks before quarantine is
        # changed.  Any malformed or incomplete reset returns the exact
        # original state.
        atomic_reset_origin = state
        state = cleared

    result = validate_axis_cycles(
        cycles, minimum_config_version=minimum_config_version
    )
    if atomic_reset_origin is not None and not result.accepted:
        return result, atomic_reset_origin
    if result.reason is FaultReason.WINDOW_TIMEOUT_FAILURE:
        first_transfer = next(
            (row for row in cycles if row.transferred), None
        )
        if first_transfer is None:
            return result, state
        metadata = StreamMetadata.unpack(first_transfer.tuser)
        return (
            result,
            dataclasses.replace(
                state,
                quarantined=True,
                quarantined_reset_epoch=metadata.reset_epoch,
                quarantined_window_id=metadata.window_id,
                poisoned_window_count=state.poisoned_window_count + 1,
            ),
        )
    if not result.accepted:
        return result, state
    transfers = transferred
    first = StreamMetadata.unpack(transfers[0].tuser)
    last = StreamMetadata.unpack(transfers[-1].tuser)
    failure: str | None = None
    if not state.initialized:
        if first.window_id != 0:
            failure = "first window in an epoch must use window_id=0"
    elif first.reset_epoch == state.reset_epoch:
        if state.last_window_id == 0xFFFFFF:
            failure = "window_id wrap requires an explicit new reset epoch"
        elif first.window_id != state.last_window_id + 1:
            failure = "window replay/gap/reorder detected across packets"
        elif first.timestamp <= state.last_final_timestamp:
            failure = "timestamp replay/regression detected across packets"
        elif first.config_version not in (
            state.config_version,
            state.config_version + 1,
        ):
            failure = "config version must hold or advance exactly one"
    elif (
        state.reset_epoch < 0xFF
        and first.reset_epoch == state.reset_epoch + 1
    ):
        if first.window_id != 0:
            failure = "new reset epoch must restart window_id at zero"
        elif first.config_version != state.config_version:
            failure = "transport reset cannot silently change bank version"
    else:
        failure = "reset epoch replay/gap/wrap detected"
    if failure is not None:
        return (
            _result(
                FaultReason.FRESHNESS_REPLAY_FAILURE,
                transfers,
                expected_count=result.expected_count,
                metadata=first,
                error_flags=result.aggregated_error_flags,
                detail=failure,
            ),
            atomic_reset_origin
            if atomic_reset_origin is not None
            else state,
        )
    return (
        result,
        IngressSequenceState(
            initialized=True,
            reset_epoch=first.reset_epoch,
            last_window_id=first.window_id,
            last_final_timestamp=last.timestamp,
            config_version=first.config_version,
            poisoned_window_count=state.poisoned_window_count,
        ),
    )


def round_shift_ties_to_even_saturate(
    value: int, *, shift: int, output_bits: int
) -> int:
    """Signed right shift with convergent rounding and signed saturation."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or isinstance(shift, bool)
        or not isinstance(shift, int)
        or shift < 0
        or isinstance(output_bits, bool)
        or not isinstance(output_bits, int)
        or output_bits < 2
    ):
        raise ValueError("invalid fixed-point conversion arguments")
    if shift:
        magnitude = abs(value)
        quotient, remainder = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and quotient & 1):
            quotient += 1
        rounded = -quotient if value < 0 else quotient
    else:
        rounded = value
    minimum = -(1 << (output_bits - 1))
    maximum = (1 << (output_bits - 1)) - 1
    return min(maximum, max(minimum, rounded))


def _round_shift_ties_to_even_unbounded(value: int, shift: int) -> int:
    if shift == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return -quotient if value < 0 else quotient


@dataclasses.dataclass(frozen=True)
class CalibrationResult:
    i_q8_16: int
    q_q8_16: int
    sticky_overflow: bool
    intermediate_i_q19_48: int
    intermediate_q_q19_48: int


@dataclasses.dataclass(frozen=True)
class MatchedFilterResult:
    accumulator_i_q16_32: int
    accumulator_q_q16_32: int
    sticky_overflow: bool
    peak_component_code: int


def _saturate_signed(value: int, bits: int) -> tuple[int, bool]:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    return min(maximum, max(minimum, value)), not minimum <= value <= maximum


def matched_filter_accumulate(
    profile_id: RateId,
    samples_q1_15: Sequence[tuple[int, int]],
    coefficient_i_q1_17: Sequence[int],
    coefficient_q_q1_17: Sequence[int],
) -> MatchedFilterResult:
    """Execute the frozen complex MAC with 35-bit terms and 48-bit sums."""

    if not isinstance(profile_id, RateId):
        raise TypeError("profile_id must be a strict rate identifier")
    try:
        count = RATE_PROFILES[RateId(profile_id)].integration_samples
    except (KeyError, ValueError):
        raise ValueError("invalid rate profile")
    if not (
        len(samples_q1_15)
        == len(coefficient_i_q1_17)
        == len(coefficient_q_q1_17)
        == count
    ):
        raise ValueError("sample and coefficient counts must equal the window")
    accumulator_i = 0
    accumulator_q = 0
    sticky = False
    peak_component = 0
    for index, ((i_code, q_code), h_i, h_q) in enumerate(
        zip(samples_q1_15, coefficient_i_q1_17, coefficient_q_q1_17)
    ):
        _require_signed(i_code, 16, f"samples_q1_15[{index}].i")
        _require_signed(q_code, 16, f"samples_q1_15[{index}].q")
        _require_signed(h_i, 18, f"coefficient_i_q1_17[{index}]")
        _require_signed(h_q, 18, f"coefficient_q_q1_17[{index}]")
        component_i = i_code * h_i + q_code * h_q
        component_q = q_code * h_i - i_code * h_q
        _require_signed(component_i, 35, "matched_filter_component_i_q3_32")
        _require_signed(component_q, 35, "matched_filter_component_q_q3_32")
        peak_component = max(
            peak_component, abs(component_i), abs(component_q)
        )
        accumulator_i, overflow_i = _saturate_signed(
            accumulator_i + component_i, 48
        )
        accumulator_q, overflow_q = _saturate_signed(
            accumulator_q + component_q, 48
        )
        sticky |= overflow_i or overflow_q
    return MatchedFilterResult(
        accumulator_i_q16_32=accumulator_i,
        accumulator_q_q16_32=accumulator_q,
        sticky_overflow=sticky,
        peak_component_code=peak_component,
    )


def _require_signed(value: int, bits: int, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -(1 << (bits - 1)) <= value < (1 << (bits - 1))
    ):
        raise ValueError(f"{name} must fit signed {bits} bits")


def calibrate_accumulators_to_q8_16(
    accumulator_i_q16_32: int,
    accumulator_q_q16_32: int,
    matrix_q2_16: tuple[int, int, int, int],
    offset_q8_16: tuple[int, int],
) -> CalibrationResult:
    """Bit-exact 2x2 affine calibration with a 67-bit Q19.48 sum."""

    _require_signed(accumulator_i_q16_32, 48, "accumulator_i_q16_32")
    _require_signed(accumulator_q_q16_32, 48, "accumulator_q_q16_32")
    if len(matrix_q2_16) != 4 or len(offset_q8_16) != 2:
        raise ValueError("calibration matrix/offset dimensions are fixed")
    for index, value in enumerate(matrix_q2_16):
        _require_signed(value, 18, f"matrix_q2_16[{index}]")
    for index, value in enumerate(offset_q8_16):
        _require_signed(value, 24, f"offset_q8_16[{index}]")
    m00, m01, m10, m11 = matrix_q2_16
    raw_i = (
        accumulator_i_q16_32 * m00
        + accumulator_q_q16_32 * m01
        + (offset_q8_16[0] << 32)
    )
    raw_q = (
        accumulator_i_q16_32 * m10
        + accumulator_q_q16_32 * m11
        + (offset_q8_16[1] << 32)
    )
    _require_signed(raw_i, 67, "calibration_i_q19_48")
    _require_signed(raw_q, 67, "calibration_q_q19_48")
    rounded_i = round_shift_ties_to_even_saturate(
        raw_i, shift=32, output_bits=24
    )
    rounded_q = round_shift_ties_to_even_saturate(
        raw_q, shift=32, output_bits=24
    )
    unsaturated_i = _round_shift_ties_to_even_unbounded(raw_i, 32)
    unsaturated_q = _round_shift_ties_to_even_unbounded(raw_q, 32)
    output_minimum = -(1 << 23)
    output_maximum = (1 << 23) - 1
    overflow = not (
        output_minimum <= unsaturated_i <= output_maximum
        and output_minimum <= unsaturated_q <= output_maximum
    )
    return CalibrationResult(
        i_q8_16=rounded_i,
        q_q8_16=rounded_q,
        sticky_overflow=overflow,
        intermediate_i_q19_48=raw_i,
        intermediate_q_q19_48=raw_q,
    )


@dataclasses.dataclass(frozen=True)
class FrontendBankPackage:
    """Serializable complete-package witness for the future A/B frontend bank."""

    schema_version: int
    config_version: int
    activation_window: int
    profile_id: int
    coefficient_i_q1_17: tuple[int, ...]
    coefficient_q_q1_17: tuple[int, ...]
    calibration_matrix_q2_16: tuple[int, int, int, int]
    calibration_offset_q8_16: tuple[int, int]
    discriminator_thresholds_q8_16: tuple[int, ...]
    threshold_qualification_state: int
    qualification_receipt_sha256: str | None = None

    def payload_bytes(self) -> bytes:
        for name in (
            "schema_version",
            "config_version",
            "activation_window",
            "profile_id",
            "threshold_qualification_state",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a strict integer")
        try:
            rate = RATE_PROFILES[RateId(self.profile_id)]
        except (KeyError, ValueError) as error:
            raise ValueError("invalid rate profile")
        if self.schema_version != 1:
            raise ValueError("unsupported schema version")
        if not 0 <= self.config_version <= 0xFFFF:
            raise ValueError("config_version must fit 16 bits")
        if not 0 <= self.activation_window <= 0xFFFFFF:
            raise ValueError("activation_window must fit 24 bits")
        if len(self.coefficient_i_q1_17) != rate.integration_samples or len(
            self.coefficient_q_q1_17
        ) != rate.integration_samples:
            raise ValueError("coefficient count must equal the exact window")
        if len(self.discriminator_thresholds_q8_16) != 4:
            raise ValueError("exactly four threshold/hysteresis codes required")
        if self.threshold_qualification_state not in (0, 1):
            raise ValueError(
                "threshold qualification state must be 0 or 1"
            )
        if self.threshold_qualification_state == 0:
            if any(self.discriminator_thresholds_q8_16):
                raise ValueError(
                    "unqualified T9.2.6 thresholds must remain exact zero"
                )
            if self.qualification_receipt_sha256 is not None:
                raise ValueError(
                    "unqualified package cannot carry a qualification receipt"
                )
            receipt = bytes(32)
        else:
            receipt_text = self.qualification_receipt_sha256
            if (
                not isinstance(receipt_text, str)
                or len(receipt_text) != 64
                or any(character not in "0123456789abcdef" for character in receipt_text)
                or receipt_text == "0" * 64
            ):
                raise ValueError(
                    "qualified package requires a nonzero lowercase SHA256 receipt"
                )
            receipt = bytes.fromhex(receipt_text)
        energy_code = sum(
            i_code * i_code + q_code * q_code
            for i_code, q_code in zip(
                self.coefficient_i_q1_17,
                self.coefficient_q_q1_17,
            )
        )
        target_energy_code = 1 << 34
        energy_tolerance_code = 1 << 22
        if abs(energy_code - target_energy_code) > energy_tolerance_code:
            raise ValueError(
                "matched-filter coefficient energy must be within 2^-12 of one"
            )
        fields: list[tuple[int, int, bool]] = [
            (self.schema_version, 16, False),
            (self.config_version, 16, False),
            (self.activation_window, 24, False),
            (self.profile_id, 8, False),
            (self.threshold_qualification_state, 8, False),
        ]
        fields.extend((value, 18, True) for value in self.coefficient_i_q1_17)
        fields.extend((value, 18, True) for value in self.coefficient_q_q1_17)
        fields.extend((value, 18, True) for value in self.calibration_matrix_q2_16)
        fields.extend((value, 24, True) for value in self.calibration_offset_q8_16)
        fields.extend(
            (value, 24, True) for value in self.discriminator_thresholds_q8_16
        )
        output = bytearray()
        for value, bits, signed in fields:
            encoded = _signed_to_twos(value, bits) if signed else value
            if not signed and not 0 <= encoded < (1 << bits):
                raise ValueError(f"value does not fit unsigned {bits} bits")
            output.extend(encoded.to_bytes((bits + 7) // 8, "little"))
        output.extend(receipt)
        return bytes(output)

    def crc32(self) -> int:
        return zlib.crc32(self.payload_bytes()) & 0xFFFFFFFF

    def sha256(self) -> str:
        return hashlib.sha256(self.payload_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class BankCommitState:
    active_bank: int = 0
    active_version: int = 0
    lkg_bank: int = 0
    lkg_version: int = 0
    active_package_sha256: str | None = None
    lkg_package_sha256: str | None = None
    active_profile_id: int | None = None
    lkg_profile_id: int | None = None
    window_open: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.active_bank, bool)
            or isinstance(self.lkg_bank, bool)
            or not isinstance(self.active_bank, int)
            or not isinstance(self.lkg_bank, int)
            or self.active_bank not in (0, 1)
            or self.lkg_bank not in (0, 1)
        ):
            raise ValueError("bank identifiers must be 0 or 1")
        if (
            isinstance(self.active_version, bool)
            or isinstance(self.lkg_version, bool)
            or not isinstance(self.active_version, int)
            or not isinstance(self.lkg_version, int)
            or not 0 <= self.active_version <= 0xFFFF
            or not 0 <= self.lkg_version <= 0xFFFF
        ):
            raise ValueError("bank versions must fit 16 bits")
        if not isinstance(self.window_open, bool):
            raise TypeError("window_open must be bool")
        for name in ("active_package_sha256", "lkg_package_sha256"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA256 or null")
        for name in ("active_profile_id", "lkg_profile_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in tuple(int(item) for item in RateId)
            ):
                raise ValueError(f"{name} must be a legal profile id or null")

    def commit(
        self,
        *,
        requested_bank: int,
        package: FrontendBankPackage,
        presented_crc32: int,
        next_window_id: int,
    ) -> "BankCommitState":
        if self.window_open:
            raise ValueError("commit is legal only between windows")
        if (
            isinstance(requested_bank, bool)
            or not isinstance(requested_bank, int)
            or requested_bank not in (0, 1)
            or requested_bank == self.active_bank
        ):
            raise ValueError("commit must target the inactive A/B bank")
        if not isinstance(package, FrontendBankPackage):
            raise TypeError("commit requires a complete FrontendBankPackage")
        if (
            isinstance(presented_crc32, bool)
            or not isinstance(presented_crc32, int)
            or not 0 <= presented_crc32 <= 0xFFFFFFFF
        ):
            raise ValueError("presented_crc32 must fit 32 bits")
        if (
            isinstance(next_window_id, bool)
            or not isinstance(next_window_id, int)
            or not 0 <= next_window_id <= 0xFFFFFF
        ):
            raise ValueError("next_window_id must fit 24 bits")
        payload = package.payload_bytes()
        computed_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
        if presented_crc32 != computed_crc32:
            raise ValueError("presented package CRC32 does not match payload")
        if package.threshold_qualification_state != 1:
            raise ValueError("unqualified frontend package cannot activate")
        if self.active_version == 0xFFFF:
            raise ValueError("version wrap is forbidden")
        if package.config_version != self.active_version + 1:
            raise ValueError("version must increase by exactly one")
        if package.activation_window != next_window_id:
            raise ValueError("activation_window must equal next_window_id")
        if (
            TRUSTED_QUALIFICATION_RECEIPT_SHA256 is None
            or package.qualification_receipt_sha256
            != TRUSTED_QUALIFICATION_RECEIPT_SHA256
        ):
            raise ValueError(
                "package qualification receipt is not the sealed trusted hash"
            )
        digest = hashlib.sha256(payload).hexdigest()
        return BankCommitState(
            active_bank=requested_bank,
            active_version=package.config_version,
            lkg_bank=self.active_bank,
            lkg_version=self.active_version,
            active_package_sha256=digest,
            lkg_package_sha256=self.active_package_sha256,
            active_profile_id=package.profile_id,
            lkg_profile_id=self.active_profile_id,
            window_open=False,
        )

    def rollback_lkg(self) -> "BankCommitState":
        if self.window_open:
            raise ValueError("rollback is legal only between windows")
        return BankCommitState(
            active_bank=self.lkg_bank,
            active_version=self.lkg_version,
            lkg_bank=self.lkg_bank,
            lkg_version=self.lkg_version,
            active_package_sha256=self.lkg_package_sha256,
            lkg_package_sha256=self.lkg_package_sha256,
            active_profile_id=self.lkg_profile_id,
            lkg_profile_id=self.lkg_profile_id,
            window_open=False,
        )


@dataclasses.dataclass(frozen=True)
class FastPathObservation:
    syndrome_code: int
    syndrome_x: str
    syndrome_z: str
    quadrature_phase_bit: int
    ood_score_code: int
    parameter_age_code: int
    reset_ack: bool = False
    observation_valid: bool = True
    deadline_ok: bool = True

    def pack_legacy_58bit_word(self) -> int:
        return encode_input_word(
            syndrome_code=self.syndrome_code,
            syndrome_x=self.syndrome_x,
            syndrome_z=self.syndrome_z,
            quadrature_phase_bit=self.quadrature_phase_bit,
            ood_score_code=self.ood_score_code,
            parameter_age_code=self.parameter_age_code,
            reset_ack=self.reset_ack,
            observation_valid=self.observation_valid,
            deadline_ok=self.deadline_ok,
        )


def fail_closed_fast_path_observation(
    candidate: FastPathObservation,
    validation: WindowValidation,
) -> FastPathObservation:
    """Map a frontend window verdict to the only legal fast-path handoff.

    An invalid window never keeps a candidate discriminator decision alive.
    The neutral codes are carried only to keep the packed schema total; the
    legacy core must inhibit the action because ``observation_valid`` is zero.
    """

    if validation.accepted:
        return candidate
    return FastPathObservation(
        syndrome_code=0,
        syndrome_x="g",
        syndrome_z="g",
        quadrature_phase_bit=0,
        ood_score_code=255,
        parameter_age_code=candidate.parameter_age_code,
        reset_ack=validation.reason is FaultReason.RESET_MID_WINDOW,
        observation_valid=False,
        deadline_ok=False,
    )


def legacy_fast_path_layout() -> dict[str, Any]:
    return INPUT_SCHEMA.to_dict()


def verify_fast_path_roundtrip(observation: FastPathObservation) -> bool:
    decoded = decode_input_word(observation.pack_legacy_58bit_word())
    return (
        decoded.input_crc_ok
        and decoded.syndrome_code == observation.syndrome_code
        and decoded.syndrome_x == observation.syndrome_x
        and decoded.syndrome_z == observation.syndrome_z
        and decoded.quadrature_phase_bit
        == observation.quadrature_phase_bit
        and decoded.ood_score_code == observation.ood_score_code
        and decoded.parameter_age_code == observation.parameter_age_code
        and decoded.reset_ack == observation.reset_ack
        and decoded.observation_valid == observation.observation_valid
        and decoded.deadline_ok == observation.deadline_ok
    )


def build_window(
    profile_id: RateId,
    *,
    domain_id: DomainId = DomainId.SYNTHETIC,
    start_timestamp: int = 0,
    window_id: int = 0,
    channel_id: int = 0,
    config_version: int = 1,
    reset_epoch: int = 0,
    iq_codes: Iterable[tuple[int, int]] | None = None,
) -> list[AxisCycle]:
    """Build an exact nominal packet for tests and future RTL cosimulation."""

    if isinstance(profile_id, bool) or not isinstance(profile_id, RateId):
        raise TypeError("profile_id must be RateId")
    if isinstance(domain_id, bool) or not isinstance(domain_id, DomainId):
        raise TypeError("domain_id must be DomainId")
    profile = RATE_PROFILES[profile_id]
    codes = (
        list(iq_codes)
        if iq_codes is not None
        else [(0, 0)] * profile.integration_samples
    )
    if len(codes) != profile.integration_samples:
        raise ValueError("iq_codes length must equal the profile window")
    output = []
    for index, (i_code, q_code) in enumerate(codes):
        metadata = StreamMetadata(
            timestamp=start_timestamp
            + index * profile.axis_cycles_per_sample,
            window_id=window_id,
            sample_index=index,
            channel_id=channel_id,
            rate_id=int(profile_id),
            domain_id=int(domain_id),
            config_version=config_version,
            reset_epoch=reset_epoch,
        )
        output.append(
            AxisCycle(
                tdata=pack_iq_tdata(i_code, q_code),
                tuser=metadata.pack(),
                tlast=index == profile.integration_samples - 1,
                tvalid=True,
                tready=True,
            )
        )
    return output


__all__ = [
    "AXIS_CLOCK_HZ",
    "AXIS_TDATA_BITS",
    "AXIS_TUSER_BITS",
    "MIN_ELASTIC_BUFFER_BEATS",
    "WINDOW_DEADLINE_AXIS_CYCLES",
    "TRUSTED_QUALIFICATION_RECEIPT_SHA256",
    "RateId",
    "RateProfile",
    "RATE_PROFILES",
    "DomainId",
    "TUSER_FIELDS",
    "ERROR_FLAG_BITS",
    "FaultReason",
    "FAULT_PRIORITY",
    "StreamMetadata",
    "AxisCycle",
    "WindowValidation",
    "pack_iq_tdata",
    "unpack_iq_tdata",
    "validate_transferred_window",
    "validate_axis_cycles",
    "IngressSequenceState",
    "validate_and_retire_sequence",
    "round_shift_ties_to_even_saturate",
    "CalibrationResult",
    "MatchedFilterResult",
    "matched_filter_accumulate",
    "calibrate_accumulators_to_q8_16",
    "FrontendBankPackage",
    "BankCommitState",
    "FastPathObservation",
    "fail_closed_fast_path_observation",
    "legacy_fast_path_layout",
    "verify_fast_path_roundtrip",
    "build_window",
]
