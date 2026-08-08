"""T5.5.1 cycle-accurate packed-word Python golden reference.

The online path accepts one CRC-protected integer input word per hardware
cycle.  It latches the active parameter image, advances the real five-stage
MAP pipeline at II=1, applies the existing health/event FSM, and publishes the
registered action one cycle later.  Parameter updates reuse the transactional
T4.3.2 A/B bank and a fixed binary image codec.

This is an RTL golden model, not RTL, synthesis, timing closure, transport, or
board evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cnn_fpga.runtime.atomic_parameter_bank import (
    AtomicParameterBankConfig,
    AtomicParameterImageBank,
    CommitAck,
    build_parameter_image_manifest,
)
from cnn_fpga.runtime.conservative_fallback import (
    HEALTH_STATUSES,
    ConservativeFallbackAction,
    ConservativeFallbackConfig,
    ConservativeFallbackController,
    ConservativeFallbackInput,
    TrustedParameterImage,
)
from cnn_fpga.runtime.experimental_event_fsm import EVENT_MODES
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTDecision,
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTPipeline,
)


MODEL_SCOPE = "packed_cycle_accurate_python_rtl_golden_not_rtl_synthesis_or_board"
TRACE_SCHEMA_VERSION = "t5.5.1-bit-accurate-hardware-trace-v1"
WORD_SCHEMA_VERSION = 0x0501
PARAMETER_IMAGE_MAGIC = b"GKP5"
PARAMETER_BUNDLE_MAGIC = b"GKPB"
PARAMETER_PHASES = 2
PARAMETER_HEADER = struct.Struct("<4sHH6BHd32s4dI4s32s")
PARAMETER_TRAILER_BYTES = 4 + 32
BUNDLE_HEADER = struct.Struct("<4sHH")

OBSERVATION_CODES = {"g": 0, "e": 1, "leakage": 2}
OBSERVATION_LABELS = {value: key for key, value in OBSERVATION_CODES.items()}
ACTION_CODES = {"I": 0, "X": 1, "Z": 2}
MODE_CODES = {name: index for index, name in enumerate(EVENT_MODES)}
HEALTH_CODES = {name: index for index, name in enumerate(HEALTH_STATUSES)}


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def crc16_ccitt_false(payload: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xffff, no reflection/xorout."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class WordField:
    name: str
    width: int
    offset: int

    @property
    def maximum(self) -> int:
        return (1 << self.width) - 1

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "width": self.width, "offset": self.offset}


def _fields(specification: Sequence[tuple[str, int]]) -> tuple[WordField, ...]:
    result = []
    offset = 0
    for name, width in specification:
        actual = _integer(width, f"{name}.width", 1)
        result.append(WordField(str(name), actual, offset))
        offset += actual
    return tuple(result)


@dataclass(frozen=True)
class PackedWordSchema:
    schema_id: str
    fields: tuple[WordField, ...]
    crc_bits: int = 16

    def __post_init__(self) -> None:
        if self.crc_bits != 16:
            raise ValueError("wire words use CRC-16/CCITT-FALSE")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("word field names must be unique")
        if any(
            field.offset != sum(item.width for item in self.fields[:index])
            for index, field in enumerate(self.fields)
        ):
            raise ValueError("word fields must be contiguous from bit zero")

    @property
    def payload_bits(self) -> int:
        return sum(field.width for field in self.fields)

    @property
    def word_bits(self) -> int:
        return self.payload_bits + self.crc_bits

    @property
    def payload_bytes(self) -> int:
        return math.ceil(self.payload_bits / 8)

    @property
    def word_hex_digits(self) -> int:
        return math.ceil(self.word_bits / 4)

    def pack(self, values: Mapping[str, int]) -> int:
        if set(values) != {field.name for field in self.fields}:
            raise ValueError("packed word values must match the schema exactly")
        payload = 0
        for field in self.fields:
            value = _integer(values[field.name], field.name)
            if value > field.maximum:
                raise ValueError(f"{field.name} exceeds its {field.width}-bit width")
            payload |= value << field.offset
        crc = crc16_ccitt_false(payload.to_bytes(self.payload_bytes, "little"))
        return payload | (crc << self.payload_bits)

    def unpack(self, word: int) -> tuple[dict[str, int], bool, int, int]:
        actual = _integer(word, "word")
        if actual >= 1 << self.word_bits:
            raise ValueError(f"word exceeds {self.word_bits}-bit schema")
        payload_mask = (1 << self.payload_bits) - 1
        payload = actual & payload_mask
        stored_crc = actual >> self.payload_bits
        expected_crc = crc16_ccitt_false(
            payload.to_bytes(self.payload_bytes, "little")
        )
        values = {
            field.name: (payload >> field.offset) & field.maximum
            for field in self.fields
        }
        return values, stored_crc == expected_crc, stored_crc, expected_crc

    def format_hex(self, word: int) -> str:
        actual = _integer(word, "word")
        if actual >= 1 << self.word_bits:
            raise ValueError("word exceeds schema")
        return f"{actual:0{self.word_hex_digits}x}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "bit_order": "little-endian field offsets; field LSB at offset",
            "byte_order_for_crc": "little-endian payload padded with zero high bits",
            "crc": "CRC-16/CCITT-FALSE poly=0x1021 init=0xffff xorout=0",
            "payload_bits": self.payload_bits,
            "crc_bits": self.crc_bits,
            "word_bits": self.word_bits,
            "fields": [field.to_dict() for field in self.fields],
        }


INPUT_SCHEMA = PackedWordSchema(
    "t5.5.1-fast-input-v1",
    _fields(
        (
            ("syndrome_code", 10),
            ("syndrome_x_code", 2),
            ("syndrome_z_code", 2),
            ("quadrature_phase_bit", 1),
            ("ood_score_code", 8),
            ("parameter_age_code", 16),
            ("reset_ack", 1),
            ("observation_valid", 1),
            ("deadline_ok", 1),
        )
    ),
)

OUTPUT_SCHEMA = PackedWordSchema(
    "t5.5.1-fast-output-v1",
    _fields(
        (
            ("output_valid", 1),
            ("mode_code", 3),
            ("correction_enable", 1),
            ("reset_request", 1),
            ("map_action_inhibited", 1),
            ("map_action_code", 2),
            ("pauli_frame_delta_x", 1),
            ("pauli_frame_delta_z", 1),
            ("pauli_frame_x", 1),
            ("pauli_frame_z", 1),
            ("phase_frame_x_code", 8),
            ("phase_frame_z_code", 8),
            ("x_e_run", 3),
            ("z_e_run", 3),
            ("leakage_run", 3),
            ("leakage_clean_run", 3),
            ("health_good_run", 3),
            ("reset_wait_run", 3),
            ("health_status_code", 3),
            ("fault_mask", 14),
            ("active_version", 16),
            ("map_llr_twos_complement", 22),
        )
    ),
)

STATE_SCHEMA = PackedWordSchema(
    "t5.5.1-fast-state-v1",
    _fields(
        (
            ("mode_code", 3),
            ("x_e_run", 3),
            ("z_e_run", 3),
            ("leakage_run", 3),
            ("leakage_clean_run", 3),
            ("health_good_run", 3),
            ("reset_wait_run", 3),
            ("pauli_frame_x", 1),
            ("pauli_frame_z", 1),
            ("phase_frame_x_code", 8),
            ("phase_frame_z_code", 8),
            ("active_version", 16),
            ("health_status_code", 3),
            ("fault_run", 8),
            ("good_run", 8),
            ("fault_cycle_count", 8),
            ("leakage_cycle_count", 8),
            *tuple((f"fault_count_{index:02d}", 8) for index in range(14)),
            ("last_fault_mask", 14),
        )
    ),
)


def encode_input_word(
    *,
    syndrome_code: int,
    syndrome_x: str,
    syndrome_z: str,
    quadrature_phase_bit: int,
    ood_score_code: int,
    parameter_age_code: int,
    reset_ack: bool = False,
    observation_valid: bool = True,
    deadline_ok: bool = True,
) -> int:
    for name, value in (("syndrome_x", syndrome_x), ("syndrome_z", syndrome_z)):
        if value not in OBSERVATION_CODES:
            raise ValueError(f"{name} must be one of {tuple(OBSERVATION_CODES)}")
    return INPUT_SCHEMA.pack(
        {
            "syndrome_code": _integer(syndrome_code, "syndrome_code"),
            "syndrome_x_code": OBSERVATION_CODES[syndrome_x],
            "syndrome_z_code": OBSERVATION_CODES[syndrome_z],
            "quadrature_phase_bit": _integer(
                quadrature_phase_bit, "quadrature_phase_bit"
            ),
            "ood_score_code": _integer(ood_score_code, "ood_score_code"),
            "parameter_age_code": _integer(
                parameter_age_code, "parameter_age_code"
            ),
            "reset_ack": int(_boolean(reset_ack, "reset_ack")),
            "observation_valid": int(
                _boolean(observation_valid, "observation_valid")
            ),
            "deadline_ok": int(_boolean(deadline_ok, "deadline_ok")),
        }
    )


@dataclass(frozen=True)
class DecodedInputWord:
    word: int
    syndrome_code: int
    syndrome_x: str
    syndrome_z: str
    quadrature_phase_bit: int
    ood_score_code: int
    parameter_age_code: int
    reset_ack: bool
    observation_valid: bool
    deadline_ok: bool
    input_crc_ok: bool
    reserved_observation_code: bool
    stored_crc16: int
    expected_crc16: int


def decode_input_word(word: int) -> DecodedInputWord:
    values, crc_ok, stored_crc, expected_crc = INPUT_SCHEMA.unpack(word)
    x_code = values["syndrome_x_code"]
    z_code = values["syndrome_z_code"]
    reserved = x_code not in OBSERVATION_LABELS or z_code not in OBSERVATION_LABELS
    return DecodedInputWord(
        word=word,
        syndrome_code=values["syndrome_code"],
        syndrome_x=OBSERVATION_LABELS.get(x_code, "g"),
        syndrome_z=OBSERVATION_LABELS.get(z_code, "g"),
        quadrature_phase_bit=values["quadrature_phase_bit"],
        ood_score_code=values["ood_score_code"],
        parameter_age_code=values["parameter_age_code"],
        reset_ack=bool(values["reset_ack"]),
        observation_valid=bool(values["observation_valid"]) and not reserved,
        deadline_ok=bool(values["deadline_ok"]),
        input_crc_ok=crc_ok,
        reserved_observation_code=reserved,
        stored_crc16=stored_crc,
        expected_crc16=expected_crc,
    )


def _signed_to_twos(value: int, width: int) -> int:
    actual = int(value)
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    if actual < minimum or actual > maximum:
        raise ValueError("signed value lies outside word width")
    return actual & ((1 << width) - 1)


def _pack_signed24(value: int) -> bytes:
    if value < -(1 << 21) or value > (1 << 21) - 1:
        raise ValueError("LLR code lies outside signed 22-bit range")
    return (value & 0xFFFFFF).to_bytes(3, "little")


def _unpack_signed24(payload: bytes) -> int:
    if len(payload) != 3:
        raise ValueError("signed24 payload must contain three bytes")
    raw = int.from_bytes(payload, "little")
    value = raw - (1 << 24) if raw & (1 << 23) else raw
    if value < -(1 << 21) or value > (1 << 21) - 1:
        raise ValueError("24-bit container is not a canonical sign extension of Q9.12")
    return value


def pack_parameter_image(image: ParametricMAPLUTImage) -> bytes:
    """Pack one selected-profile image into a fixed little-endian binary layout."""

    if not isinstance(image, ParametricMAPLUTImage):
        raise TypeError("image must be ParametricMAPLUTImage")
    image.verify()
    config = image.config
    selected = ParametricMAPLUTConfig()
    if config != selected:
        raise ValueError("T5.5.1 binary codec accepts the selected fixed-point profile only")
    header = PARAMETER_HEADER.pack(
        PARAMETER_IMAGE_MAGIC,
        WORD_SCHEMA_VERSION,
        image.active_bank_version,
        config.adc_bits,
        config.address_bits,
        config.fraction_bits,
        config.llr_integer_bits,
        config.llr_fractional_bits,
        PARAMETER_PHASES,
        config.table_entries,
        config.lattice,
        bytes.fromhex(image.source_params_sha256),
        image.model_mean[0],
        image.model_mean[1],
        image.model_sigma[0],
        image.model_sigma[1],
        image.llr_saturation_count,
        bytes.fromhex(image.image_crc32),
        bytes.fromhex(image.image_sha256),
    )
    tables = b"".join(
        _pack_signed24(code) for table in image.table_codes for code in table
    )
    body = header + tables
    return (
        body
        + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
        + hashlib.sha256(body).digest()
    )


def unpack_parameter_image(payload: bytes) -> ParametricMAPLUTImage:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    selected = ParametricMAPLUTConfig()
    table_bytes = PARAMETER_PHASES * selected.table_entries * 3
    expected_length = PARAMETER_HEADER.size + table_bytes + PARAMETER_TRAILER_BYTES
    if len(payload) != expected_length:
        raise ValueError(
            f"packed parameter image must contain exactly {expected_length} bytes"
        )
    body = payload[:-PARAMETER_TRAILER_BYTES]
    stored_crc = struct.unpack("<I", payload[-PARAMETER_TRAILER_BYTES:-32])[0]
    stored_sha = payload[-32:]
    if stored_crc != zlib.crc32(body) & 0xFFFFFFFF:
        raise ValueError("packed parameter image CRC32 mismatch")
    if stored_sha != hashlib.sha256(body).digest():
        raise ValueError("packed parameter image SHA256 mismatch")
    unpacked = PARAMETER_HEADER.unpack(body[: PARAMETER_HEADER.size])
    (
        magic,
        schema,
        version,
        adc_bits,
        address_bits,
        fraction_bits,
        llr_integer_bits,
        llr_fractional_bits,
        phases,
        entries,
        lattice,
        source_sha,
        mean_q,
        mean_p,
        sigma_q,
        sigma_p,
        saturation_count,
        image_crc,
        image_sha,
    ) = unpacked
    if magic != PARAMETER_IMAGE_MAGIC or schema != WORD_SCHEMA_VERSION:
        raise ValueError("packed parameter image magic/schema mismatch")
    if (
        adc_bits,
        address_bits,
        fraction_bits,
        llr_integer_bits,
        llr_fractional_bits,
        phases,
        entries,
    ) != (
        selected.adc_bits,
        selected.address_bits,
        selected.fraction_bits,
        selected.llr_integer_bits,
        selected.llr_fractional_bits,
        PARAMETER_PHASES,
        selected.table_entries,
    ) or lattice != selected.lattice:
        raise ValueError("packed parameter image fixed configuration mismatch")
    cursor = PARAMETER_HEADER.size
    tables = []
    for _ in range(PARAMETER_PHASES):
        table = []
        for _ in range(entries):
            table.append(_unpack_signed24(body[cursor : cursor + 3]))
            cursor += 3
        tables.append(tuple(table))
    image = ParametricMAPLUTImage(
        config=selected,
        active_bank_version=version,
        source_params_sha256=source_sha.hex(),
        model_mean=(mean_q, mean_p),
        model_sigma=(sigma_q, sigma_p),
        table_codes=(tables[0], tables[1]),
        llr_saturation_count=saturation_count,
        image_crc32=image_crc.hex(),
        image_sha256=image_sha.hex(),
    )
    image.verify()
    if pack_parameter_image(image) != payload:
        raise ValueError("packed parameter image is not canonical")
    return image


def pack_parameter_bundle(images: Sequence[ParametricMAPLUTImage]) -> bytes:
    registered = tuple(images)
    if not registered:
        raise ValueError("parameter bundle must contain at least one image")
    if [image.active_bank_version for image in registered] != list(range(len(registered))):
        raise ValueError("parameter bundle versions must be contiguous from zero")
    parts = [BUNDLE_HEADER.pack(PARAMETER_BUNDLE_MAGIC, WORD_SCHEMA_VERSION, len(registered))]
    for image in registered:
        payload = pack_parameter_image(image)
        parts.extend((struct.pack("<I", len(payload)), payload))
    body = b"".join(parts)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF) + hashlib.sha256(body).digest()


def unpack_parameter_bundle(payload: bytes) -> tuple[ParametricMAPLUTImage, ...]:
    if not isinstance(payload, bytes) or len(payload) < BUNDLE_HEADER.size + PARAMETER_TRAILER_BYTES:
        raise ValueError("parameter bundle is truncated")
    body = payload[:-PARAMETER_TRAILER_BYTES]
    stored_crc = struct.unpack("<I", payload[-PARAMETER_TRAILER_BYTES:-32])[0]
    if stored_crc != zlib.crc32(body) & 0xFFFFFFFF or payload[-32:] != hashlib.sha256(body).digest():
        raise ValueError("parameter bundle integrity mismatch")
    magic, schema, count = BUNDLE_HEADER.unpack(body[: BUNDLE_HEADER.size])
    if magic != PARAMETER_BUNDLE_MAGIC or schema != WORD_SCHEMA_VERSION:
        raise ValueError("parameter bundle magic/schema mismatch")
    cursor = BUNDLE_HEADER.size
    images = []
    for _ in range(count):
        if cursor + 4 > len(body):
            raise ValueError("parameter bundle length prefix is truncated")
        length = struct.unpack("<I", body[cursor : cursor + 4])[0]
        cursor += 4
        if cursor + length > len(body):
            raise ValueError("parameter bundle image is truncated")
        images.append(unpack_parameter_image(body[cursor : cursor + length]))
        cursor += length
    if cursor != len(body):
        raise ValueError("parameter bundle has trailing bytes")
    if [image.active_bank_version for image in images] != list(range(count)):
        raise ValueError("parameter bundle versions are not contiguous")
    if pack_parameter_bundle(images) != payload:
        raise ValueError("parameter bundle is not canonical")
    return tuple(images)


@dataclass(frozen=True)
class _LatchedRequest:
    source_cycle: int
    decoded: DecodedInputWord
    image: ParametricMAPLUTImage


@dataclass(frozen=True)
class _ScheduledOutput:
    source_cycle: int
    decision: ParametricMAPLUTDecision
    action: ConservativeFallbackAction


@dataclass(frozen=True)
class HardwareTraceRecord:
    trace_schema_version: str
    hardware_cycle: int
    input_valid: bool
    input_word_hex: str
    input_crc_ok: bool | None
    input_latched_version: int | None
    map_valid: bool
    map_source_cycle: int | None
    map_valid_cycle: int | None
    map_active_version: int | None
    map_address: int | None
    map_fraction_code: int | None
    map_llr_code: int | None
    output_valid: bool
    output_source_cycle: int | None
    output_word_hex: str
    output_crc_ok: bool
    state_word_hex: str
    state_crc_ok: bool
    active_bank: str
    active_version: int
    commit_status: str
    commit_reason: str
    safe_boundary: bool
    trace_chain_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop("trace_chain_sha256")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class BitAccurateHardwareReference:
    """Packed-word, true-pipeline, transactional-bank golden reference."""

    def __init__(
        self,
        images: Sequence[ParametricMAPLUTImage],
        *,
        bank_config: AtomicParameterBankConfig | None = None,
        max_parameter_age_cycles: int = 64,
    ) -> None:
        registered = tuple(images)
        if not registered:
            raise ValueError("images must not be empty")
        if [image.active_bank_version for image in registered] != list(range(len(registered))):
            raise ValueError("images must have contiguous versions from zero")
        for image in registered:
            image.verify()
            if image.config != ParametricMAPLUTConfig():
                raise ValueError("hardware reference requires the selected fixed profile")
        self.images = registered
        self.bank = AtomicParameterImageBank(registered[0], bank_config)
        self.pipeline = ParametricMAPLUTPipeline(registered[0])
        trusted = tuple(
            TrustedParameterImage(
                image.active_bank_version, image.image_crc32, image.image_sha256
            )
            for image in registered
        )
        self.controller = ConservativeFallbackController(
            trusted,
            ConservativeFallbackConfig(
                initial_active_bank_version=0,
                max_parameter_age_cycles=max_parameter_age_cycles,
            ),
        )
        self._cycle = 0
        self._metadata: dict[int, _LatchedRequest] = {}
        self._outputs: dict[int, _ScheduledOutput] = {}
        self._latest_action: ConservativeFallbackAction | None = None
        self._trace: list[HardwareTraceRecord] = []
        self._chain = bytes(32)
        self._draining = False

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def trace(self) -> tuple[HardwareTraceRecord, ...]:
        return tuple(self._trace)

    @property
    def final_trace_sha256(self) -> str:
        return self._chain.hex()

    def stage_packed_update(
        self,
        packed_image: bytes,
        *,
        transaction_id: str,
        selection_key: str,
        source_window_id: int,
        created_cycle: int,
        apply_cycle: int,
        chunk_bytes: int = 257,
    ) -> dict[str, Any]:
        image = unpack_parameter_image(packed_image)
        if image.active_bank_version >= len(self.images) or image != self.images[image.active_bank_version]:
            raise ValueError("packed update is not in the frozen image registry")
        if image.active_bank_version != self.bank.active_version + 1:
            raise ValueError("packed update version must be active+1")
        first_window = source_window_id - 1
        if first_window < 1:
            raise ValueError("source_window_id must allow two hysteresis windows")
        self.bank.observe_selection(
            window_id=first_window, selection_key=selection_key, eligible=True
        )
        self.bank.observe_selection(
            window_id=source_window_id, selection_key=selection_key, eligible=True
        )
        manifest, canonical_payload = build_parameter_image_manifest(
            image,
            transaction_id=transaction_id,
            selection_key=selection_key,
            expected_active_version=self.bank.active_version,
            source_window_id=source_window_id,
            created_epoch=created_cycle,
            apply_epoch=apply_cycle,
        )
        self.bank.begin_stage(manifest, current_epoch=created_cycle)
        chunk = _integer(chunk_bytes, "chunk_bytes", 1)
        for offset in range(0, len(canonical_payload), chunk):
            self.bank.write_chunk(
                transaction_id,
                offset=offset,
                chunk=canonical_payload[offset : offset + chunk],
            )
        staged = self.bank.finalize_stage(
            transaction_id, current_epoch=created_cycle
        )
        return {
            "packed_bytes": len(packed_image),
            "packed_crc32": f"{zlib.crc32(packed_image) & 0xFFFFFFFF:08x}",
            "packed_sha256": hashlib.sha256(packed_image).hexdigest(),
            "canonical_transfer_bytes": len(canonical_payload),
            "target_bank": staged.target_bank,
            "new_version": manifest.new_version,
            "apply_cycle": manifest.apply_epoch,
        }

    def _pack_output(self, scheduled: _ScheduledOutput | None) -> int:
        if scheduled is None:
            return OUTPUT_SCHEMA.pack(
                {field.name: 0 for field in OUTPUT_SCHEMA.fields}
            )
        hardware = scheduled.action.hardware_action
        decision = scheduled.decision
        return OUTPUT_SCHEMA.pack(
            {
                "output_valid": 1,
                "mode_code": MODE_CODES[hardware.mode],
                "correction_enable": int(hardware.correction_enable),
                "reset_request": int(hardware.reset_request),
                "map_action_inhibited": int(hardware.map_action_inhibited),
                "map_action_code": ACTION_CODES[hardware.map_logical_action],
                "pauli_frame_delta_x": int(hardware.pauli_frame_delta_x),
                "pauli_frame_delta_z": int(hardware.pauli_frame_delta_z),
                "pauli_frame_x": int(hardware.pauli_frame_x),
                "pauli_frame_z": int(hardware.pauli_frame_z),
                "phase_frame_x_code": hardware.phase_frame_x_code,
                "phase_frame_z_code": hardware.phase_frame_z_code,
                "x_e_run": hardware.x_e_run,
                "z_e_run": hardware.z_e_run,
                "leakage_run": hardware.leakage_run,
                "leakage_clean_run": hardware.leakage_clean_run,
                "health_good_run": hardware.health_good_run,
                "reset_wait_run": hardware.reset_wait_run,
                "health_status_code": HEALTH_CODES[scheduled.action.status],
                "fault_mask": scheduled.action.fault_mask,
                "active_version": hardware.active_bank_version,
                "map_llr_twos_complement": _signed_to_twos(decision.llr_code, 22),
            }
        )

    def _pack_state(self) -> int:
        controller = self.controller.state
        hardware = None if self._latest_action is None else self._latest_action.hardware_action
        values = {
            "mode_code": 0 if hardware is None else MODE_CODES[hardware.mode],
            "x_e_run": 0 if hardware is None else hardware.x_e_run,
            "z_e_run": 0 if hardware is None else hardware.z_e_run,
            "leakage_run": 0 if hardware is None else hardware.leakage_run,
            "leakage_clean_run": 0 if hardware is None else hardware.leakage_clean_run,
            "health_good_run": 0 if hardware is None else hardware.health_good_run,
            "reset_wait_run": 0 if hardware is None else hardware.reset_wait_run,
            "pauli_frame_x": 0 if hardware is None else int(hardware.pauli_frame_x),
            "pauli_frame_z": 0 if hardware is None else int(hardware.pauli_frame_z),
            "phase_frame_x_code": 0 if hardware is None else hardware.phase_frame_x_code,
            "phase_frame_z_code": 0 if hardware is None else hardware.phase_frame_z_code,
            "active_version": controller.trusted_active_bank_version,
            "health_status_code": HEALTH_CODES[controller.status],
            "fault_run": controller.fault_run,
            "good_run": controller.good_run,
            "fault_cycle_count": controller.fault_cycle_count,
            "leakage_cycle_count": controller.leakage_cycle_count,
            **{
                f"fault_count_{index:02d}": controller.per_flag_cycle_counts[index]
                for index in range(14)
            },
            "last_fault_mask": controller.last_fault_mask,
        }
        return STATE_SCHEMA.pack(values)

    def step_word(
        self, input_word: int | None, *, safe_boundary: bool = True
    ) -> HardwareTraceRecord:
        safe = _boolean(safe_boundary, "safe_boundary")
        cycle = self._cycle
        emitted = self._outputs.pop(cycle, None)
        output_word = self._pack_output(emitted)
        _, output_crc_ok, _, _ = OUTPUT_SCHEMA.unpack(output_word)

        ack = self.bank.commit_if_ready(cycle, safe_boundary=safe)
        if ack is not None and ack.accepted:
            self.pipeline.load_image(self.bank.read_active_image())

        decoded = None
        request = None
        latched_version = None
        if input_word is None:
            self._draining = True
        else:
            if self._draining:
                raise ValueError("new input is forbidden after drain has started")
            decoded = decode_input_word(input_word)
            image = self.bank.read_active_image()
            latched_version = image.active_bank_version
            request = ParametricMAPLUTInput(
                cycle,
                decoded.syndrome_code,
                decoded.quadrature_phase_bit,
                image.active_bank_version,
            )
            self._metadata[cycle] = _LatchedRequest(cycle, decoded, image)

        decision = self.pipeline.step(cycle, request)
        if decision is not None:
            latched = self._metadata.pop(decision.input_cycle)
            fallback = self.controller.step(
                ConservativeFallbackInput(
                    cycle_index=cycle,
                    syndrome_x=latched.decoded.syndrome_x,
                    syndrome_z=latched.decoded.syndrome_z,
                    quadrature_phase_bit=latched.decoded.quadrature_phase_bit,
                    map_decision=decision,
                    expected_active_bank_version=latched.image.active_bank_version,
                    reported_image_crc32=latched.image.image_crc32,
                    reported_image_sha256=latched.image.image_sha256,
                    parameter_age_cycles=latched.decoded.parameter_age_code,
                    ood_score_code=latched.decoded.ood_score_code,
                    reset_ack=latched.decoded.reset_ack,
                    observation_valid=latched.decoded.observation_valid,
                    input_crc_ok=latched.decoded.input_crc_ok,
                    deadline_ok=latched.decoded.deadline_ok,
                )
            )
            self._latest_action = fallback
            action_cycle = fallback.hardware_action.action_cycle
            if action_cycle in self._outputs:
                raise RuntimeError("output register collision violates II=1")
            self._outputs[action_cycle] = _ScheduledOutput(
                decision.input_cycle, decision, fallback
            )

        state_word = self._pack_state()
        _, state_crc_ok, _, _ = STATE_SCHEMA.unpack(state_word)
        unsigned = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "hardware_cycle": cycle,
            "input_valid": input_word is not None,
            "input_word_hex": "" if input_word is None else INPUT_SCHEMA.format_hex(input_word),
            "input_crc_ok": None if decoded is None else decoded.input_crc_ok,
            "input_latched_version": latched_version,
            "map_valid": decision is not None,
            "map_source_cycle": None if decision is None else decision.input_cycle,
            "map_valid_cycle": None if decision is None else decision.valid_cycle,
            "map_active_version": None if decision is None else decision.active_bank_version,
            "map_address": None if decision is None else decision.address,
            "map_fraction_code": None if decision is None else decision.fraction_code,
            "map_llr_code": None if decision is None else decision.llr_code,
            "output_valid": emitted is not None,
            "output_source_cycle": None if emitted is None else emitted.source_cycle,
            "output_word_hex": OUTPUT_SCHEMA.format_hex(output_word),
            "output_crc_ok": output_crc_ok,
            "state_word_hex": STATE_SCHEMA.format_hex(state_word),
            "state_crc_ok": state_crc_ok,
            "active_bank": self.bank.active_bank,
            "active_version": self.bank.active_version,
            "commit_status": "none" if ack is None else ack.status,
            "commit_reason": "" if ack is None else ack.reason,
            "safe_boundary": safe,
        }
        self._chain = hashlib.sha256(self._chain + _canonical_bytes(unsigned)).digest()
        record = HardwareTraceRecord(
            **unsigned, trace_chain_sha256=self._chain.hex()
        )
        self._trace.append(record)
        self._cycle += 1
        return record


def hardware_reference_contract() -> dict[str, Any]:
    selected = ParametricMAPLUTConfig()
    return {
        "model_scope": MODEL_SCOPE,
        "clock_semantics": (
            "publish prior output; atomically commit A/B image; latch input and image; "
            "advance MAP/FSM; register action for next cycle"
        ),
        "map_pipeline_cycles": selected.pipeline_latency_cycles,
        "event_output_register_cycles": 1,
        "source_to_output_cycles": selected.pipeline_latency_cycles + 1,
        "initiation_interval_cycles": selected.initiation_interval_cycles,
        "rounding": {
            "ADC": "upstream unsigned code; float replay is outside online reference",
            "LUT_interpolation": "signed integer right shift round-to-nearest ties-to-even",
            "logical_action": "strict llr_code < 0",
            "counters": "unsigned saturation with no wrap",
            "Pauli_frame": "GF(2) XOR",
            "phase_frame": "modulo-256 half-turn addition",
        },
        "input_word": INPUT_SCHEMA.to_dict(),
        "output_word": OUTPUT_SCHEMA.to_dict(),
        "state_word": STATE_SCHEMA.to_dict(),
        "parameter_image": {
            "schema_version": WORD_SCHEMA_VERSION,
            "header_bytes": PARAMETER_HEADER.size,
            "table_container_bits": 24,
            "logical_llr_bits": selected.llr_word_bits,
            "phase_tables": PARAMETER_PHASES,
            "entries_per_phase": selected.table_entries,
            "trailer": "CRC32 then SHA256 over header+tables",
            "packed_bytes_per_image": len(
                pack_parameter_image(
                    ParametricMAPLUTImage.create(
                        config=selected,
                        active_bank_version=0,
                        source_params_sha256="0" * 64,
                        model_mean=(0.0, 0.0),
                        model_sigma=(1.0, 1.0),
                        table_codes=(
                            (0,) * selected.table_entries,
                            (0,) * selected.table_entries,
                        ),
                        llr_saturation_count=0,
                    )
                )
            ),
            "binary_runtime_float_operations": 0,
        },
        "parameter_bank": {
            "banks": 2,
            "stage_target": "inactive bank only after complete CRC/SHA/image verification",
            "commit": "CAS/version/hysteresis/residency/safe-boundary atomic switch",
            "inflight_rule": "requests latch image/version at S0; commit never changes pending decisions",
        },
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "hardware_fields": {
            "rtl_generated": False,
            "synthesized": False,
            "fmax_mhz": None,
            "target_lut_count": None,
            "target_ff_count": None,
            "target_bram_count": None,
            "target_dsp_count": None,
            "board_measured": False,
        },
    }


__all__ = [
    "ACTION_CODES",
    "BitAccurateHardwareReference",
    "DecodedInputWord",
    "HardwareTraceRecord",
    "INPUT_SCHEMA",
    "MODEL_SCOPE",
    "OUTPUT_SCHEMA",
    "STATE_SCHEMA",
    "TRACE_SCHEMA_VERSION",
    "WORD_SCHEMA_VERSION",
    "crc16_ccitt_false",
    "decode_input_word",
    "encode_input_word",
    "hardware_reference_contract",
    "pack_parameter_bundle",
    "pack_parameter_image",
    "unpack_parameter_bundle",
    "unpack_parameter_image",
]
