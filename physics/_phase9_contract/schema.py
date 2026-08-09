"""Schemas, finite encodings, and immutable values for the Phase-9 contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from typing import Any, Iterable, Mapping, Sequence

MODEL_SCOPE = {
    "kind": "EXECUTABLE_CAUSAL_INTERFACE_AND_TOTAL_RECURRENCE",
    "physics_backend_qualified": False,
    "codebook_released": False,
    "frontend_released": False,
    "rtl_adapter_qualified": False,
    "performance_evaluated": False,
}

CLAIM_BOUNDARY = {
    "allowed": (
        "finite observed-only interface contract; exhaustive factorised "
        "map totality; deterministic fail-closed semantics"
    ),
    "forbidden": [
        "high_fidelity_twin_qualified",
        "final_recovery_codebook",
        "current_rtl_wire_compatibility",
        "measured_fpga_or_hil",
        "logical_error_rate_or_lifetime_gain",
        "official_or_puviani_surpass",
        "external_sota_or_rank",
    ],
}


def _field(
    type_name: str,
    *,
    required: bool = True,
    bits: int | None = None,
    description: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": type_name,
        "required": required,
        "description": description,
    }
    if bits is not None:
        value["bits"] = bits
    return value


NAMESPACE_SCHEMAS: dict[str, dict[str, Any]] = {
    "BACKEND_LATENT": {
        "deployable": False,
        "additional_properties": False,
        "fields": {
            "oscillator_shift_q": _field(
                "float64", description="hidden oscillator q displacement"
            ),
            "oscillator_shift_p": _field(
                "float64", description="hidden oscillator p displacement"
            ),
            "oscillator_loss_state": _field(
                "float64", description="hidden loss/channel state"
            ),
            "ancilla_level": _field(
                "enum[g,e,f,higher]", description="hidden multilevel ancilla"
            ),
            "leakage_age": _field(
                "uint16", bits=16, description="hidden leakage residence age"
            ),
            "readout_state": _field(
                "opaque", description="hidden readout-chain state"
            ),
            "reset_state": _field(
                "opaque", description="hidden reset-channel state"
            ),
            "calibration_state": _field(
                "opaque", description="hidden device calibration"
            ),
            "drift_regime": _field(
                "opaque", description="hidden drift regime"
            ),
            "measurement_backaction_state": _field(
                "opaque", description="post-measurement hidden state"
            ),
        },
    },
    "DEPLOYABLE_OBSERVED": {
        "deployable": True,
        "additional_properties": False,
        "fields": {
            "iq_i": _field(
                "fixed_array[int]",
                description="digitised or replayed I samples; never truth",
            ),
            "iq_q": _field(
                "fixed_array[int]",
                description="digitised or replayed Q samples; never truth",
            ),
            "iq_source": _field(
                "enum[synthetic,recorded,live]",
                description="typed acquisition domain",
            ),
            "matched_filter_i": _field(
                "fixed_int", description="causal matched-filter I output"
            ),
            "matched_filter_q": _field(
                "fixed_int", description="causal matched-filter Q output"
            ),
            "llr_q": _field(
                "fixed_int", description="causal fixed-point q LLR"
            ),
            "llr_p": _field(
                "fixed_int", description="causal fixed-point p LLR"
            ),
            "discriminator_word": _field(
                "uint8", bits=8, description="finite fast-path word"
            ),
            "discriminator_confidence": _field(
                "uint8", bits=8, description="quantised confidence"
            ),
            "leakage_confidence": _field(
                "uint8",
                bits=8,
                description="observed leakage evidence, never truth",
            ),
            "timestamp": _field(
                "uint64", bits=64, description="monotonic source timestamp"
            ),
            "reset_ack": _field(
                "enum[none,success,failure]",
                description="observed physical reset outcome",
            ),
            "previous_action_word": _field(
                "uint80", bits=80, description="previous, never current, action"
            ),
            "previous_composite_key": _field(
                "uint24",
                bits=24,
                description=(
                    "canonical prior K receipt used to recompute and "
                    "authenticate the previous action deterministically"
                ),
            ),
            "previous_action_present": _field(
                "bool", description="whether prior action receipt is valid"
            ),
            "previous_active_image_version": _field(
                "uint16",
                bits=16,
                description="nondecision version receipt sideband",
            ),
        },
    },
    "CONTROLLER_MEMORY": {
        "deployable": True,
        "additional_properties": False,
        "fields": {
            "bank_id": _field(
                "uint2", bits=2, description="A/B/LKG/invalid handle"
            ),
            "image_version": _field(
                "uint16", bits=16, description="monotonic package version"
            ),
            "trusted_version": _field(
                "uint16", bits=16, description="last-known-good version"
            ),
            "phase_frame": _field(
                "uint2", bits=2, description="finite q/p phase frame"
            ),
            "pauli_frame": _field(
                "uint2", bits=2, description="finite logical Pauli frame"
            ),
            "previous_event_class": _field(
                "uint4",
                bits=4,
                description="prior event receipt; current event is derived",
            ),
            "leakage_reset_fsm_state": _field(
                "uint8", bits=8, description="finite safety FSM state"
            ),
            "integrity_flags": _field(
                "uint16",
                bits=16,
                description="frozen raw fault bits; current event is derived",
            ),
        },
    },
    "EVALUATOR_TRUTH": {
        "deployable": False,
        "additional_properties": False,
        "fields": {
            "logical_state": _field(
                "opaque", description="six-state evaluator truth"
            ),
            "logical_error": _field(
                "bool", description="evaluation-only logical error"
            ),
            "hidden_trajectory": _field(
                "opaque", description="evaluation-only latent trajectory"
            ),
            "counterfactual_outcomes": _field(
                "opaque", description="evaluation-only counterfactual truth"
            ),
        },
    },
    "PROVENANCE": {
        "deployable": False,
        "additional_properties": False,
        "fields": {
            "backend_id": _field(
                "string", description="auditable backend identity"
            ),
            "seed_id": _field(
                "string", description="auditable exogenous seed identity"
            ),
            "trace_sha256": _field(
                "sha256", description="immutable trajectory binding"
            ),
            "config_sha256": _field(
                "sha256", description="immutable configuration binding"
            ),
            "code_revision": _field(
                "string", description="auditable source revision"
            ),
        },
    },
}

TRUTH_PROVENANCE_DENYLIST = (
    "BACKEND_LATENT",
    "EVALUATOR_TRUTH",
    "PROVENANCE",
    "logical_error",
    "logical_state",
    "hidden_trajectory",
    "counterfactual",
    "future",
    "seed_id",
    "trace_sha256",
    "config_sha256",
)

DISCRIMINATOR_LAYOUT = {
    "total_bits": 8,
    "fields": [
        {"field": "axis", "lsb": 7, "bits": 1},
        {"field": "quantized_class", "lsb": 1, "bits": 6},
        {"field": "reserved_ood", "lsb": 0, "bits": 1},
    ],
    "legal_rule": "reserved_ood==0 and quantized_class<63",
    "legal_count": 126,
    "ood_or_reserved_count": 130,
    "confidence_is_separate_observed_field": True,
}

PHASE_FRAME_SEMANTICS = {
    "logical_bits": 2,
    "states": {
        "0": {"q_byte": 0, "p_byte": 0},
        "1": {"q_byte": 128, "p_byte": 0},
        "2": {"q_byte": 0, "p_byte": 128},
        "3": {"q_byte": 128, "p_byte": 128},
    },
    "current_rtl_two_uint8_adapter_qualified": False,
}

FSM_ENCODING = {
    "bits": 8,
    "mode_bits": 3,
    "active_dwell_counter_bits": 5,
    "valid_modes": [
        "NORMAL",
        "HOLD",
        "LEAKAGE",
        "RESETTING",
        "RECOVERING",
        "FAULT",
    ],
    "reserved_mode_codes": [6, 7],
    "current_rtl_six_counter_adapter_qualified": False,
    "interpretation": (
        "new Markov safety-state proposal; not a claim of bit compatibility "
        "with the existing experimental_event_fsm"
    ),
}


class BankId(IntEnum):
    A = 0
    B = 1
    LKG = 2
    INVALID = 3


class EventClass(IntEnum):
    NORMAL = 0
    Q_POSITIVE = 1
    Q_NEGATIVE = 2
    P_POSITIVE = 3
    P_NEGATIVE = 4
    LOW_CONFIDENCE = 5
    LEAKAGE = 6
    RESET_ACK_SUCCESS = 7
    RESET_ACK_FAILURE = 8
    OOD = 9
    INTEGRITY_FAULT = 10
    VERSION_FAULT = 11
    DEADLINE_FAULT = 12
    RESERVED_13 = 13
    RESERVED_14 = 14
    RESERVED_15 = 15


class FsmMode(IntEnum):
    NORMAL = 0
    HOLD = 1
    LEAKAGE = 2
    RESETTING = 3
    RECOVERING = 4
    FAULT = 5


class NominalAction(IntEnum):
    IDLE = 0
    X = 1
    Z = 2
    XZ = 3
    RESET = 4
    HOLD = 5
    LKG_HOLD = 6
    INVALID = 7


class ReasonCode(IntEnum):
    NOMINAL = 0
    LOW_CONFIDENCE = 1
    LEAKAGE_DETECTED = 2
    RESET_SUCCESS = 3
    RESET_FAILURE = 4
    OOD_WORD = 5
    INTEGRITY_FAULT = 6
    VERSION_FAULT = 7
    DEADLINE_FAULT = 8
    INVALID_BANK = 9
    INVALID_DISCRIMINATOR = 10
    INVALID_FSM = 11
    RESERVED_EVENT = 12
    LKG_ACTIVE = 13
    RECOVERY_HYSTERESIS = 14
    PERSISTENT_LEAKAGE = 15


ACTION_LAYOUT = [
    {"field": "valid", "bits": 1, "lsb": 0},
    {"field": "action_code", "bits": 3, "lsb": 1},
    {"field": "correction_enable", "bits": 1, "lsb": 4},
    {"field": "reset_request", "bits": 1, "lsb": 5},
    {"field": "fallback", "bits": 1, "lsb": 6},
    {"field": "hold", "bits": 1, "lsb": 7},
    {"field": "pauli_dx", "bits": 1, "lsb": 8},
    {"field": "pauli_dz", "bits": 1, "lsb": 9},
    {"field": "next_phase_frame", "bits": 2, "lsb": 10},
    {"field": "next_fsm_state", "bits": 8, "lsb": 12},
    {"field": "catalog_action_id", "bits": 6, "lsb": 20},
    {"field": "residual_q", "bits": 1, "lsb": 26},
    {"field": "residual_p", "bits": 1, "lsb": 27},
    {"field": "reason_code", "bits": 6, "lsb": 28},
    {"field": "error_flags", "bits": 8, "lsb": 34},
    {"field": "source_bank_id", "bits": 2, "lsb": 42},
    {"field": "factor_tag", "bits": 16, "lsb": 44},
    {"field": "reserved_zero", "bits": 4, "lsb": 60},
    {"field": "crc16", "bits": 16, "lsb": 64},
]

CRC16_CONTRACT = {
    "name": "CRC-16/CCITT-FALSE",
    "polynomial": 0x1021,
    "initial_value": 0xFFFF,
    "xor_out": 0x0000,
    "reflect_input": False,
    "reflect_output": False,
    "payload_byte_order": "little",
    "crc_field_position": "bits_64_through_79",
}

ACTION_SIDEBAND_CONTRACT = {
    "logical_action_word_bits": 80,
    "active_image_version_bits": 16,
    "active_image_version_is_decision_input": False,
    "active_image_version_is_nondecision_receipt_sideband": True,
    "old_or_new_complete_image_latched_per_accepted_request": True,
    "current_118bit_rtl_adapter_qualified": False,
}

OBSERVATION_ENVELOPE_BOUNDARY = {
    "t9_2_1_transport_neutral_checks": {
        "iq_nonempty": True,
        "iq_lengths_equal": True,
        "maximum_samples_per_frame": 65536,
        "sample_container_bound": "signed_int64",
    },
    "sample_rate": None,
    "integration_window_samples": None,
    "adc_sample_bits": None,
    "coefficient_bits": None,
    "accumulator_bits": None,
    "llr_bits": None,
    "q_format": None,
    "rounding": None,
    "saturation": None,
    "deferred_to": "T9.2.6",
}

INTEGRITY_FLAG_LAYOUT = {
    "INPUT_CRC": 0,
    "IMAGE_CRC": 1,
    "IMAGE_SHA": 2,
    "UNKNOWN_VERSION": 3,
    "VERSION_MISMATCH": 4,
    "ROLLBACK_VERSION": 5,
    "STALE_PACKAGE": 6,
    "PARTIAL_PACKAGE": 7,
    "DEADLINE_MISS": 8,
    "RESET_ACK_UNEXPECTED": 9,
}

SLOW_PATH_BOUNDARY = {
    "validator_scope": "COMPLETE_TRUSTED_NOMINATION_ONLY",
    "active_bank_write_authority": False,
    "single_entry_patch_authority": False,
    "per_cycle_action_authority": False,
    "requires_trusted_schema_registry": True,
    "requires_trusted_package_ledger": True,
    "requires_trusted_provenance": True,
    "requires_external_release_pin": True,
    "requires_safe_activation_epoch": True,
    "atomic_bank_integration_qualified": False,
    "downstream_qualification": ["T9.3.3", "T9.5.1", "T9.5.3"],
}

PAYLOAD_BITS = 64
ACTION_WORD_BITS = 80
NOMINAL_CELL_COUNT = 4 * 256
TRANSITION_CELL_COUNT = 4 * 16 * 256 * 8

FAULT_PRIORITY = (
    "INVALID_KEY",
    "RESERVED_ACTION_OR_EVENT",
    "INPUT_CRC",
    "IMAGE_CRC",
    "IMAGE_SHA",
    "PARTIAL_PACKAGE",
    "UNKNOWN_VERSION",
    "VERSION_MISMATCH",
    "ROLLBACK_VERSION",
    "STALE_PACKAGE",
    "DEADLINE_MISS",
    "OOD_WORD",
    "RESET_ACK_UNEXPECTED",
    "PERSISTENT_LEAKAGE",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def crc16_ccitt(payload: bytes) -> int:
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def is_legal_discriminator_word(word: int) -> bool:
    if type(word) is not int or not 0 <= word <= 0xFF:
        return False
    reserved = word & 0x1
    quantized_class = (word >> 1) & 0x3F
    return reserved == 0 and quantized_class < 63


def legal_discriminator_words() -> tuple[int, ...]:
    return tuple(word for word in range(256) if is_legal_discriminator_word(word))


def encode_fsm(mode: FsmMode | int, counter: int) -> int:
    mode_value = int(mode)
    if mode_value not in set(int(item) for item in FsmMode):
        raise ValueError("invalid FSM mode")
    if type(counter) is not int or not 0 <= counter <= 31:
        raise ValueError("FSM counter must be an exact integer in [0,31]")
    return (mode_value << 5) | counter


def decode_fsm(state: int) -> tuple[FsmMode, int]:
    if type(state) is not int or not 0 <= state <= 0xFF:
        raise ValueError("FSM state must be an exact uint8")
    mode_value = state >> 5
    if mode_value not in set(int(item) for item in FsmMode):
        raise ValueError("reserved FSM encoding")
    return FsmMode(mode_value), state & 0x1F


def _contains_denied_token(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            text = str(key).casefold()
            if any(token.casefold() in text for token in TRUTH_PROVENANCE_DENYLIST):
                return True
            if _contains_denied_token(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_denied_token(item) for item in value)
    if isinstance(value, str):
        text = value.casefold()
        return any(token.casefold() in text for token in TRUTH_PROVENANCE_DENYLIST)
    return False


def validate_namespace(
    namespace_id: str,
    payload: Mapping[str, Any],
    *,
    require_all_fields: bool = True,
) -> None:
    if namespace_id not in NAMESPACE_SCHEMAS:
        raise ValueError(f"unknown namespace: {namespace_id}")
    if not isinstance(payload, Mapping):
        raise TypeError("namespace payload must be a mapping")
    schema = NAMESPACE_SCHEMAS[namespace_id]
    allowed = set(schema["fields"])
    actual = set(payload)
    if actual - allowed:
        raise ValueError(f"additional fields are forbidden: {sorted(actual - allowed)}")
    required = {
        name
        for name, descriptor in schema["fields"].items()
        if descriptor["required"]
    }
    if require_all_fields and actual != required:
        missing = sorted(required - actual)
        raise ValueError(f"missing required fields: {missing}")
    for name in actual:
        if not _schema_value_ok(payload[name], schema["fields"][name]):
            raise TypeError(
                f"{namespace_id}.{name} violates frozen type "
                f"{schema['fields'][name]['type']}"
            )


def _schema_value_ok(value: Any, descriptor: Mapping[str, Any]) -> bool:
    type_name = descriptor["type"]
    bits = descriptor.get("bits")
    if bits is not None and type_name.startswith(("uint", "fixed_int")):
        return type(value) is int and 0 <= value < (1 << int(bits))
    if type_name.startswith("uint"):
        suffix = type_name.removeprefix("uint")
        width = int(suffix) if suffix.isdigit() else int(bits or 0)
        return (
            width > 0
            and type(value) is int
            and 0 <= value < (1 << width)
        )
    if type_name == "fixed_int":
        return type(value) is int
    if type_name == "fixed_array[int]":
        return isinstance(value, (list, tuple)) and all(
            type(item) is int for item in value
        )
    if type_name == "float64":
        return type(value) in {int, float}
    if type_name == "bool":
        return type(value) is bool
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "sha256":
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
        )
    if type_name == "opaque":
        return value is not None
    if type_name.startswith("enum[") and type_name.endswith("]"):
        allowed = type_name[5:-1].split(",")
        return isinstance(value, str) and value in allowed
    return False

def _exact_uint(value: Any, bits: int, field: str) -> int:
    if (
        type(value) is bool
        or not isinstance(value, int)
        or not 0 <= int(value) < (1 << bits)
    ):
        raise ValueError(f"{field} must be an exact uint{bits}")
    return int(value)


@dataclass(frozen=True)
class CompositeKey:
    bank_id: int
    discriminator_word: int
    phase_frame: int
    event_class: int
    leakage_reset_fsm_state: int

    def __post_init__(self) -> None:
        _exact_uint(self.bank_id, 2, "bank_id")
        _exact_uint(self.discriminator_word, 8, "discriminator_word")
        _exact_uint(self.phase_frame, 2, "phase_frame")
        _exact_uint(self.event_class, 4, "event_class")
        _exact_uint(
            self.leakage_reset_fsm_state,
            8,
            "leakage_reset_fsm_state",
        )

    def to_word(self) -> int:
        """Pack the exact five-field logical K into a 24-bit receipt."""

        return (
            self.bank_id
            | (self.discriminator_word << 2)
            | (self.phase_frame << 10)
            | (self.event_class << 12)
            | (self.leakage_reset_fsm_state << 16)
        )

    @classmethod
    def from_word(cls, word: int) -> "CompositeKey":
        packed = _exact_uint(word, 24, "packed_composite_key")
        return cls(
            bank_id=packed & 0x3,
            discriminator_word=(packed >> 2) & 0xFF,
            phase_frame=(packed >> 10) & 0x3,
            event_class=(packed >> 12) & 0xF,
            leakage_reset_fsm_state=(packed >> 16) & 0xFF,
        )


@dataclass(frozen=True)
class IntegrityStatus:
    input_crc_ok: bool = True
    image_crc_ok: bool = True
    image_sha_ok: bool = True
    version_known: bool = True
    version_matches: bool = True
    no_version_rollback: bool = True
    package_fresh: bool = True
    package_complete: bool = True
    deadline_met: bool = True
    reset_ack_expected: bool = True

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if type(value) is not bool:
                raise TypeError("integrity flags must be exact bool values")

    def first_fault(self) -> str | None:
        ordered = (
            ("INPUT_CRC", self.input_crc_ok),
            ("IMAGE_CRC", self.image_crc_ok),
            ("IMAGE_SHA", self.image_sha_ok),
            ("PARTIAL_PACKAGE", self.package_complete),
            ("UNKNOWN_VERSION", self.version_known),
            ("VERSION_MISMATCH", self.version_matches),
            ("ROLLBACK_VERSION", self.no_version_rollback),
            ("STALE_PACKAGE", self.package_fresh),
            ("DEADLINE_MISS", self.deadline_met),
            ("RESET_ACK_UNEXPECTED", self.reset_ack_expected),
        )
        return next((name for name, passed in ordered if not passed), None)


@dataclass(frozen=True)
class NominalResult:
    action_index: int
    reason_code: int
    error_flags: int

    def __post_init__(self) -> None:
        _exact_uint(self.action_index, 3, "action_index")
        _exact_uint(self.reason_code, 6, "reason_code")
        _exact_uint(self.error_flags, 8, "error_flags")

    def to_bytes(self) -> bytes:
        return bytes((self.action_index, self.reason_code, self.error_flags))


@dataclass(frozen=True)
class TransitionResult:
    action_code: int
    correction_enable: bool
    reset_request: bool
    fallback: bool
    hold: bool
    pauli_dx: int
    pauli_dz: int
    next_phase_frame: int
    next_fsm_state: int
    catalog_action_id: int
    reason_code: int
    error_flags: int

    def __post_init__(self) -> None:
        _exact_uint(self.action_code, 3, "action_code")
        for name in ("correction_enable", "reset_request", "fallback", "hold"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        _exact_uint(self.pauli_dx, 1, "pauli_dx")
        _exact_uint(self.pauli_dz, 1, "pauli_dz")
        _exact_uint(self.next_phase_frame, 2, "next_phase_frame")
        _exact_uint(self.next_fsm_state, 8, "next_fsm_state")
        decode_fsm(self.next_fsm_state)
        _exact_uint(self.catalog_action_id, 6, "catalog_action_id")
        _exact_uint(self.reason_code, 6, "reason_code")
        if int(self.reason_code) not in {int(item) for item in ReasonCode}:
            raise ValueError("reserved reason code must never be emitted")
        _exact_uint(self.error_flags, 8, "error_flags")

    def to_bytes(self) -> bytes:
        flags = (
            int(self.correction_enable)
            | (int(self.reset_request) << 1)
            | (int(self.fallback) << 2)
            | (int(self.hold) << 3)
        )
        return bytes(
            (
                self.action_code,
                flags,
                self.pauli_dx | (self.pauli_dz << 1),
                self.next_phase_frame,
                self.next_fsm_state,
                self.catalog_action_id,
                self.reason_code,
                self.error_flags,
            )
        )


@dataclass(frozen=True)
class ActionWord:
    action_code: int
    correction_enable: bool
    reset_request: bool
    fallback: bool
    hold: bool
    pauli_dx: int
    pauli_dz: int
    next_phase_frame: int
    next_fsm_state: int
    catalog_action_id: int
    reason_code: int
    error_flags: int
    source_bank_id: int
    factor_tag: int
    residual_q: int = 0
    residual_p: int = 0
    valid: int = 1

    def __post_init__(self) -> None:
        _exact_uint(self.valid, 1, "valid")
        if self.valid != 1:
            raise ValueError("action word valid bit must be one")
        _exact_uint(self.action_code, 3, "action_code")
        if self.action_code == NominalAction.INVALID:
            raise ValueError("reserved action code must never be emitted")
        for name in ("correction_enable", "reset_request", "fallback", "hold"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        _exact_uint(self.pauli_dx, 1, "pauli_dx")
        _exact_uint(self.pauli_dz, 1, "pauli_dz")
        _exact_uint(self.next_phase_frame, 2, "next_phase_frame")
        _exact_uint(self.next_fsm_state, 8, "next_fsm_state")
        decode_fsm(self.next_fsm_state)
        _exact_uint(self.catalog_action_id, 6, "catalog_action_id")
        _exact_uint(self.reason_code, 6, "reason_code")
        if int(self.reason_code) not in {int(item) for item in ReasonCode}:
            raise ValueError("reserved reason code must never be emitted")
        _exact_uint(self.error_flags, 8, "error_flags")
        _exact_uint(self.source_bank_id, 2, "source_bank_id")
        _exact_uint(self.factor_tag, 16, "factor_tag")
        _exact_uint(self.residual_q, 1, "residual_q")
        _exact_uint(self.residual_p, 1, "residual_p")
        if self.residual_q != 0 or self.residual_p != 0:
            raise ValueError(
                "base-lane residual is structurally fixed to bit-exact zero"
            )
        code = int(self.action_code)
        correction_expected = code in {
            int(NominalAction.X),
            int(NominalAction.Z),
            int(NominalAction.XZ),
        }
        if self.correction_enable is not correction_expected:
            raise ValueError("action_code/correction_enable contradiction")
        if self.reset_request is not (code == int(NominalAction.RESET)):
            raise ValueError("action_code/reset_request contradiction")
        if self.hold is not (
            code
            in {
                int(NominalAction.HOLD),
                int(NominalAction.LKG_HOLD),
            }
        ):
            raise ValueError("action_code/hold contradiction")
        expected_dx, expected_dz = _nominal_pauli(code)
        if (self.pauli_dx, self.pauli_dz) != (
            expected_dx,
            expected_dz,
        ):
            raise ValueError("action_code/Pauli delta contradiction")
        if self.correction_enable and self.fallback:
            raise ValueError("fallback cannot emit a correction")
        if self.fallback and code not in {
            int(NominalAction.RESET),
            int(NominalAction.HOLD),
            int(NominalAction.LKG_HOLD),
        }:
            raise ValueError("fallback action must be reset or hold")
        if code == int(NominalAction.RESET):
            mode, _ = decode_fsm(self.next_fsm_state)
            if (
                not self.fallback
                or mode != FsmMode.RESETTING
                or int(self.reason_code)
                not in {
                    int(ReasonCode.RESET_FAILURE),
                    int(ReasonCode.RECOVERY_HYSTERESIS),
                    int(ReasonCode.PERSISTENT_LEAKAGE),
                }
                or not (self.error_flags & 0x80)
            ):
                raise ValueError(
                    "RESET receipt violates reachable fail-closed semantics"
                )

    def payload(self) -> int:
        values = {
            "valid": self.valid,
            "action_code": self.action_code,
            "correction_enable": int(self.correction_enable),
            "reset_request": int(self.reset_request),
            "fallback": int(self.fallback),
            "hold": int(self.hold),
            "pauli_dx": self.pauli_dx,
            "pauli_dz": self.pauli_dz,
            "next_phase_frame": self.next_phase_frame,
            "next_fsm_state": self.next_fsm_state,
            "catalog_action_id": self.catalog_action_id,
            "residual_q": self.residual_q,
            "residual_p": self.residual_p,
            "reason_code": self.reason_code,
            "error_flags": self.error_flags,
            "source_bank_id": self.source_bank_id,
            "factor_tag": self.factor_tag,
            "reserved_zero": 0,
        }
        payload = 0
        for field in ACTION_LAYOUT[:-1]:
            value = values[field["field"]]
            if value >= 1 << field["bits"]:
                raise ValueError(f"{field['field']} exceeds frozen width")
            payload |= value << field["lsb"]
        if payload >= 1 << PAYLOAD_BITS:
            raise AssertionError("payload layout exceeds 64 bits")
        return payload

    def pack(self) -> int:
        payload = self.payload()
        crc = crc16_ccitt(payload.to_bytes(8, "little"))
        return payload | (crc << PAYLOAD_BITS)

    def to_bytes(self) -> bytes:
        return self.pack().to_bytes(10, "little")

    @classmethod
    def unpack(cls, packed: int) -> "ActionWord":
        _exact_uint(packed, ACTION_WORD_BITS, "packed_action_word")
        payload = packed & ((1 << PAYLOAD_BITS) - 1)
        encoded_crc = packed >> PAYLOAD_BITS
        expected_crc = crc16_ccitt(payload.to_bytes(8, "little"))
        if encoded_crc != expected_crc:
            raise ValueError("action-word CRC16 mismatch")
        values: dict[str, int] = {}
        for field in ACTION_LAYOUT[:-1]:
            values[field["field"]] = (
                payload >> field["lsb"]
            ) & ((1 << field["bits"]) - 1)
        if values["reserved_zero"] != 0:
            raise ValueError("reserved action bits must be zero")
        return cls(
            valid=values["valid"],
            action_code=values["action_code"],
            correction_enable=bool(values["correction_enable"]),
            reset_request=bool(values["reset_request"]),
            fallback=bool(values["fallback"]),
            hold=bool(values["hold"]),
            pauli_dx=values["pauli_dx"],
            pauli_dz=values["pauli_dz"],
            next_phase_frame=values["next_phase_frame"],
            next_fsm_state=values["next_fsm_state"],
            catalog_action_id=values["catalog_action_id"],
            residual_q=values["residual_q"],
            residual_p=values["residual_p"],
            reason_code=values["reason_code"],
            error_flags=values["error_flags"],
            source_bank_id=values["source_bank_id"],
            factor_tag=values["factor_tag"],
        )


@dataclass(frozen=True)
class RecurrenceResult:
    action_word: ActionWord
    next_phase_frame: int
    next_fsm_state: int
    reason_code: int
    error_flags: int
    fault_id: str | None

def _nominal_pauli(action: int) -> tuple[int, int]:
    return (
        int(action in (NominalAction.X, NominalAction.XZ)),
        int(action in (NominalAction.Z, NominalAction.XZ)),
    )

