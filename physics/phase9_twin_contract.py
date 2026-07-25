"""Executable T9.2.1 causal digital-twin interface contract.

This module deliberately freezes *interfaces* rather than claiming that a
high-fidelity physics backend, a recovery codebook, or a hardware adapter has
already been qualified.  It supplies four things that a prose-only schema
cannot:

* strict, disjoint latent/observed/memory/truth/provenance namespaces;
* a finite factorisation ``F(K)=T(phase,event,fsm,N(bank,word))``;
* a CRC-protected 80-bit logical action word whose base residual is
  structurally zero; and
* exhaustive, reproducible enumeration of every N and T cell.

The logical action word is a Phase-9 contract.  It is not the existing
58/118/232-bit RTL wire format; T9.2.6 and T9.5 must qualify an adapter before
any hardware claim is possible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
import hashlib
import inspect
import json
from typing import Any, Iterable, Mapping, Sequence
import zlib


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


def validate_deployable_inputs(
    observed: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> None:
    validate_namespace("DEPLOYABLE_OBSERVED", observed)
    validate_namespace("CONTROLLER_MEMORY", memory)
    if _contains_denied_token(observed) or _contains_denied_token(memory):
        raise ValueError("hidden, future, truth, or provenance token rejected recursively")
    iq_i = observed["iq_i"]
    iq_q = observed["iq_q"]
    if not iq_i or len(iq_i) != len(iq_q):
        raise ValueError("I/Q frames must be nonempty and equal length")
    maximum = OBSERVATION_ENVELOPE_BOUNDARY[
        "t9_2_1_transport_neutral_checks"
    ]["maximum_samples_per_frame"]
    if len(iq_i) > maximum:
        raise ValueError("I/Q frame exceeds transport-neutral maximum")
    signed_limit = 1 << 63
    if any(
        not -signed_limit <= sample < signed_limit
        for sample in (*iq_i, *iq_q)
    ):
        raise ValueError("I/Q sample exceeds signed-int64 container bound")
    receipt_present = observed["previous_action_present"]
    previous_word = observed["previous_action_word"]
    previous_key_word = observed["previous_composite_key"]
    previous_version = observed["previous_active_image_version"]
    if not receipt_present:
        if (
            previous_word != 0
            or previous_key_word != 0
            or previous_version != 0
        ):
            raise ValueError(
                "absent previous-action receipt must use zero sentinels"
            )
    else:
        previous_key = CompositeKey.from_word(previous_key_word)
        previous = ActionWord.unpack(previous_word)
        expected = total_recurrence(previous_key).action_word
        if previous != expected:
            raise ValueError(
                "previous action is not the exact recurrence of its "
                "canonical key receipt"
            )
        if previous.next_phase_frame != memory["phase_frame"]:
            raise ValueError("previous action/current phase receipt mismatch")
        if (
            previous.next_fsm_state
            != memory["leakage_reset_fsm_state"]
        ):
            raise ValueError("previous action/current FSM receipt mismatch")


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


def nominal_cell(bank_id: int, discriminator_word: int) -> NominalResult:
    _exact_uint(bank_id, 2, "bank_id")
    _exact_uint(discriminator_word, 8, "discriminator_word")
    if bank_id == BankId.INVALID:
        return NominalResult(
            NominalAction.INVALID, ReasonCode.INVALID_BANK, 0x01
        )
    if bank_id == BankId.LKG:
        return NominalResult(
            NominalAction.LKG_HOLD, ReasonCode.LKG_ACTIVE, 0x01
        )
    if not is_legal_discriminator_word(discriminator_word):
        return NominalResult(
            NominalAction.LKG_HOLD,
            ReasonCode.OOD_WORD,
            0x02,
        )
    quantized_class = (discriminator_word >> 1) & 0x3F
    axis = (discriminator_word >> 7) & 0x1
    if quantized_class <= 8:
        action = NominalAction.IDLE
    elif axis == 0 and quantized_class <= 35:
        action = NominalAction.X
    elif axis == 1 and quantized_class <= 35:
        action = NominalAction.Z
    else:
        action = NominalAction.XZ
    return NominalResult(action, ReasonCode.NOMINAL, 0)


def _phase_after_action(phase_frame: int, action: int) -> int:
    if action == NominalAction.X:
        return phase_frame ^ 0b01
    if action == NominalAction.Z:
        return phase_frame ^ 0b10
    if action == NominalAction.XZ:
        return phase_frame ^ 0b11
    return phase_frame


def _nominal_pauli(action: int) -> tuple[int, int]:
    return (
        int(action in (NominalAction.X, NominalAction.XZ)),
        int(action in (NominalAction.Z, NominalAction.XZ)),
    )


def transition_cell(
    phase_frame: int,
    event_class: int,
    leakage_reset_fsm_state: int,
    nominal_action_index: int,
) -> TransitionResult:
    _exact_uint(phase_frame, 2, "phase_frame")
    _exact_uint(event_class, 4, "event_class")
    _exact_uint(
        leakage_reset_fsm_state, 8, "leakage_reset_fsm_state"
    )
    _exact_uint(nominal_action_index, 3, "nominal_action_index")

    try:
        mode, counter = decode_fsm(leakage_reset_fsm_state)
    except ValueError:
        return TransitionResult(
            action_code=NominalAction.LKG_HOLD,
            correction_enable=False,
            reset_request=False,
            fallback=True,
            hold=True,
            pauli_dx=0,
            pauli_dz=0,
            next_phase_frame=phase_frame,
            next_fsm_state=encode_fsm(FsmMode.FAULT, 0),
            catalog_action_id=0,
            reason_code=ReasonCode.INVALID_FSM,
            error_flags=0x81,
        )

    event = EventClass(event_class)
    if nominal_action_index == NominalAction.INVALID:
        return TransitionResult(
            NominalAction.LKG_HOLD,
            False,
            False,
            True,
            True,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.HOLD, min(counter + 1, 31)),
            0,
            ReasonCode.INVALID_DISCRIMINATOR,
            0x01,
        )
    if event in {
        EventClass.RESERVED_13,
        EventClass.RESERVED_14,
        EventClass.RESERVED_15,
    }:
        return TransitionResult(
            NominalAction.LKG_HOLD,
            False,
            False,
            True,
            True,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.FAULT, 0),
            0,
            ReasonCode.RESERVED_EVENT,
            0x81,
        )

    if event in {
        EventClass.OOD,
        EventClass.INTEGRITY_FAULT,
        EventClass.VERSION_FAULT,
        EventClass.DEADLINE_FAULT,
    }:
        reason = {
            EventClass.OOD: ReasonCode.OOD_WORD,
            EventClass.INTEGRITY_FAULT: ReasonCode.INTEGRITY_FAULT,
            EventClass.VERSION_FAULT: ReasonCode.VERSION_FAULT,
            EventClass.DEADLINE_FAULT: ReasonCode.DEADLINE_FAULT,
        }[event]
        return TransitionResult(
            NominalAction.LKG_HOLD,
            False,
            False,
            True,
            True,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.HOLD, min(counter + 1, 31)),
            0,
            reason,
            0x02,
        )

    if nominal_action_index == NominalAction.LKG_HOLD:
        return TransitionResult(
            NominalAction.LKG_HOLD,
            False,
            False,
            True,
            True,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.HOLD, min(counter + 1, 31)),
            0,
            ReasonCode.LKG_ACTIVE,
            0x01,
        )

    if event == EventClass.LEAKAGE:
        next_counter = min(counter + 1, 31)
        persistent = next_counter >= 3
        return TransitionResult(
            NominalAction.RESET if persistent else NominalAction.HOLD,
            False,
            persistent,
            True,
            not persistent,
            0,
            0,
            phase_frame,
            encode_fsm(
                FsmMode.RESETTING if persistent else FsmMode.LEAKAGE,
                next_counter,
            ),
            0,
            (
                ReasonCode.PERSISTENT_LEAKAGE
                if persistent
                else ReasonCode.LEAKAGE_DETECTED
            ),
            0x80,
        )

    if event == EventClass.RESET_ACK_FAILURE:
        return TransitionResult(
            NominalAction.RESET,
            False,
            True,
            True,
            False,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.RESETTING, min(counter + 1, 31)),
            0,
            ReasonCode.RESET_FAILURE,
            0x80,
        )

    if event == EventClass.RESET_ACK_SUCCESS:
        return TransitionResult(
            NominalAction.HOLD,
            False,
            False,
            True,
            True,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.RECOVERING, 0),
            0,
            ReasonCode.RESET_SUCCESS,
            0,
        )

    if event == EventClass.LOW_CONFIDENCE:
        return TransitionResult(
            NominalAction.HOLD,
            False,
            False,
            True,
            True,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.HOLD, min(counter + 1, 31)),
            0,
            ReasonCode.LOW_CONFIDENCE,
            0,
        )

    if mode in {FsmMode.LEAKAGE, FsmMode.RESETTING, FsmMode.FAULT}:
        return TransitionResult(
            NominalAction.RESET,
            False,
            True,
            True,
            False,
            0,
            0,
            phase_frame,
            encode_fsm(FsmMode.RESETTING, min(counter + 1, 31)),
            0,
            ReasonCode.RECOVERY_HYSTERESIS,
            0x80,
        )
    if mode in {FsmMode.HOLD, FsmMode.RECOVERING}:
        next_counter = min(counter + 1, 31)
        recovered = next_counter >= 2
        return TransitionResult(
            NominalAction.HOLD,
            False,
            False,
            True,
            True,
            0,
            0,
            phase_frame,
            encode_fsm(
                FsmMode.NORMAL if recovered else FsmMode.RECOVERING,
                0 if recovered else next_counter,
            ),
            0,
            ReasonCode.RECOVERY_HYSTERESIS,
            0,
        )

    pauli_dx, pauli_dz = _nominal_pauli(nominal_action_index)
    next_phase = _phase_after_action(phase_frame, nominal_action_index)
    return TransitionResult(
        nominal_action_index,
        nominal_action_index
        in {NominalAction.X, NominalAction.Z, NominalAction.XZ},
        nominal_action_index == NominalAction.RESET,
        False,
        nominal_action_index == NominalAction.HOLD,
        pauli_dx,
        pauli_dz,
        next_phase,
        encode_fsm(FsmMode.NORMAL, 0),
        nominal_action_index,
        ReasonCode.NOMINAL,
        0,
    )


@dataclass(frozen=True)
class CanonicalizedKey:
    key: CompositeKey
    source_fault_id: str | None


@dataclass(frozen=True)
class KeyAssembly:
    key: CompositeKey
    source_fault_id: str | None
    observed_timestamp: int


def classify_raw_fault(
    key: CompositeKey,
    integrity: IntegrityStatus,
) -> str | None:
    """Apply the single frozen priority order to simultaneous faults."""

    try:
        decode_fsm(key.leakage_reset_fsm_state)
        fsm_valid = True
    except ValueError:
        fsm_valid = False
    conditions = {
        "INVALID_KEY": (
            key.bank_id == BankId.INVALID
            or not fsm_valid
        ),
        "RESERVED_ACTION_OR_EVENT": key.event_class
        in {
            EventClass.RESERVED_13,
            EventClass.RESERVED_14,
            EventClass.RESERVED_15,
        },
        "INPUT_CRC": not integrity.input_crc_ok,
        "IMAGE_CRC": not integrity.image_crc_ok,
        "IMAGE_SHA": not integrity.image_sha_ok,
        "PARTIAL_PACKAGE": not integrity.package_complete,
        "UNKNOWN_VERSION": not integrity.version_known,
        "VERSION_MISMATCH": not integrity.version_matches,
        "ROLLBACK_VERSION": not integrity.no_version_rollback,
        "STALE_PACKAGE": not integrity.package_fresh,
        "DEADLINE_MISS": not integrity.deadline_met,
        "OOD_WORD": (
            key.event_class == EventClass.OOD
            or not is_legal_discriminator_word(
                key.discriminator_word
            )
        ),
        "RESET_ACK_UNEXPECTED": not integrity.reset_ack_expected,
        "PERSISTENT_LEAKAGE": False,
    }
    if fsm_valid and key.event_class == EventClass.LEAKAGE:
        _, counter = decode_fsm(key.leakage_reset_fsm_state)
        conditions["PERSISTENT_LEAKAGE"] = counter >= 2
    return next(
        (fault for fault in FAULT_PRIORITY if conditions[fault]),
        None,
    )


def canonicalize_composite_key(
    key: CompositeKey,
    integrity: IntegrityStatus,
) -> CanonicalizedKey:
    """Map raw integrity/version status into the finite event-class field.

    This function is upstream of ``F``.  Distinct raw faults that require the
    same bounded response intentionally collapse to one canonical event.  Once
    the key is returned, the recurrence has no hidden side input.
    """

    if not isinstance(key, CompositeKey):
        raise TypeError("key must be a CompositeKey")
    if not isinstance(integrity, IntegrityStatus):
        raise TypeError("integrity must be an IntegrityStatus")
    fault = classify_raw_fault(key, integrity)
    if fault is None or fault in {
        "INVALID_KEY",
        "RESERVED_ACTION_OR_EVENT",
        "OOD_WORD",
        "PERSISTENT_LEAKAGE",
    }:
        return CanonicalizedKey(key, fault)
    event_by_fault = {
        "INPUT_CRC": EventClass.INTEGRITY_FAULT,
        "IMAGE_CRC": EventClass.INTEGRITY_FAULT,
        "IMAGE_SHA": EventClass.INTEGRITY_FAULT,
        "PARTIAL_PACKAGE": EventClass.INTEGRITY_FAULT,
        "UNKNOWN_VERSION": EventClass.VERSION_FAULT,
        "VERSION_MISMATCH": EventClass.VERSION_FAULT,
        "ROLLBACK_VERSION": EventClass.VERSION_FAULT,
        "STALE_PACKAGE": EventClass.VERSION_FAULT,
        "DEADLINE_MISS": EventClass.DEADLINE_FAULT,
        "RESET_ACK_UNEXPECTED": EventClass.RESET_ACK_FAILURE,
    }
    canonical = CompositeKey(
        bank_id=key.bank_id,
        discriminator_word=key.discriminator_word,
        phase_frame=key.phase_frame,
        event_class=int(event_by_fault[fault]),
        leakage_reset_fsm_state=key.leakage_reset_fsm_state,
    )
    return CanonicalizedKey(canonical, fault)


def _integrity_from_flags(flags: int) -> IntegrityStatus:
    _exact_uint(flags, 16, "integrity_flags")
    if flags >> 10:
        raise ValueError("reserved integrity flag bits must be zero")
    bad = {
        name: bool(flags & (1 << bit))
        for name, bit in INTEGRITY_FLAG_LAYOUT.items()
    }
    return IntegrityStatus(
        input_crc_ok=not bad["INPUT_CRC"],
        image_crc_ok=not bad["IMAGE_CRC"],
        image_sha_ok=not bad["IMAGE_SHA"],
        version_known=not bad["UNKNOWN_VERSION"],
        version_matches=not bad["VERSION_MISMATCH"],
        no_version_rollback=not bad["ROLLBACK_VERSION"],
        package_fresh=not bad["STALE_PACKAGE"],
        package_complete=not bad["PARTIAL_PACKAGE"],
        deadline_met=not bad["DEADLINE_MISS"],
        reset_ack_expected=not bad["RESET_ACK_UNEXPECTED"],
    )


def _event_from_observed(observed: Mapping[str, Any]) -> EventClass:
    reset_ack = observed["reset_ack"]
    if reset_ack == "success":
        return EventClass.RESET_ACK_SUCCESS
    if reset_ack == "failure":
        return EventClass.RESET_ACK_FAILURE
    if observed["leakage_confidence"] >= 224:
        return EventClass.LEAKAGE
    if observed["discriminator_confidence"] < 32:
        return EventClass.LOW_CONFIDENCE
    word = observed["discriminator_word"]
    if not is_legal_discriminator_word(word):
        return EventClass.OOD
    quantized_class = (word >> 1) & 0x3F
    axis = (word >> 7) & 0x1
    if quantized_class <= 8 or quantized_class > 35:
        return EventClass.NORMAL
    positive = quantized_class >= 22
    if axis == 0:
        return (
            EventClass.Q_POSITIVE
            if positive
            else EventClass.Q_NEGATIVE
        )
    return (
        EventClass.P_POSITIVE
        if positive
        else EventClass.P_NEGATIVE
    )


def assemble_deployable_key(
    observed: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> KeyAssembly:
    """The only deployable observed+memory -> canonical-key assembler."""

    validate_deployable_inputs(observed, memory)
    raw = CompositeKey(
        bank_id=memory["bank_id"],
        discriminator_word=observed["discriminator_word"],
        phase_frame=memory["phase_frame"],
        event_class=int(_event_from_observed(observed)),
        leakage_reset_fsm_state=memory[
            "leakage_reset_fsm_state"
        ],
    )
    integrity = _integrity_from_flags(memory["integrity_flags"])
    rollback_ok = integrity.no_version_rollback
    if observed["previous_action_present"]:
        rollback_ok = rollback_ok and (
            memory["image_version"]
            >= observed["previous_active_image_version"]
        )
    version_matches = (
        integrity.version_matches
        and memory["trusted_version"] <= memory["image_version"]
    )
    reset_ack_expected = integrity.reset_ack_expected
    if observed["reset_ack"] != "none":
        if not observed["previous_action_present"]:
            reset_ack_expected = False
        else:
            previous = ActionWord.unpack(
                observed["previous_action_word"]
            )
            reset_ack_expected = (
                reset_ack_expected and previous.reset_request
            )
    integrity = IntegrityStatus(
        input_crc_ok=integrity.input_crc_ok,
        image_crc_ok=integrity.image_crc_ok,
        image_sha_ok=integrity.image_sha_ok,
        version_known=integrity.version_known,
        version_matches=version_matches,
        no_version_rollback=rollback_ok,
        package_fresh=integrity.package_fresh,
        package_complete=integrity.package_complete,
        deadline_met=integrity.deadline_met,
        reset_ack_expected=reset_ack_expected,
    )
    canonical = canonicalize_composite_key(
        raw, integrity
    )
    return KeyAssembly(
        key=canonical.key,
        source_fault_id=canonical.source_fault_id,
        observed_timestamp=observed["timestamp"],
    )


def _factorized_fault_id(
    key: CompositeKey,
    nominal: NominalResult,
) -> str | None:
    try:
        _, counter = decode_fsm(key.leakage_reset_fsm_state)
    except ValueError:
        return "INVALID_KEY"
    if nominal.action_index == NominalAction.INVALID:
        return "INVALID_KEY"
    if nominal.reason_code == ReasonCode.OOD_WORD:
        return "OOD_WORD"
    event = EventClass(key.event_class)
    if event in {
        EventClass.RESERVED_13,
        EventClass.RESERVED_14,
        EventClass.RESERVED_15,
    }:
        return "RESERVED_ACTION_OR_EVENT"
    if event == EventClass.INTEGRITY_FAULT:
        return "INTEGRITY_FAULT_CLASS"
    if event == EventClass.VERSION_FAULT:
        return "VERSION_FAULT_CLASS"
    if event == EventClass.DEADLINE_FAULT:
        return "DEADLINE_MISS"
    if event == EventClass.OOD:
        return "OOD_WORD"
    if event == EventClass.LEAKAGE and counter >= 2:
        return "PERSISTENT_LEAKAGE"
    return None


def _compose_reason_error(
    nominal: NominalResult,
    transition: TransitionResult,
) -> tuple[int, int]:
    reason = transition.reason_code
    if (
        nominal.reason_code != ReasonCode.NOMINAL
        and transition.reason_code
        in {
            ReasonCode.INVALID_DISCRIMINATOR,
            ReasonCode.LKG_ACTIVE,
        }
    ):
        reason = nominal.reason_code
    return int(reason), transition.error_flags | nominal.error_flags


def _factor_tag(
    bank_id: int,
    phase_frame: int,
    event_class: int,
    fsm_state: int,
    nominal_action_index: int,
) -> int:
    payload = bytes(
        (
            _exact_uint(bank_id, 2, "bank_id"),
            _exact_uint(phase_frame, 2, "phase_frame"),
            _exact_uint(event_class, 4, "event_class"),
            _exact_uint(fsm_state, 8, "fsm_state"),
            _exact_uint(
                nominal_action_index, 3, "nominal_action_index"
            ),
        )
    )
    return crc16_ccitt(payload)


def total_recurrence(key: CompositeKey) -> RecurrenceResult:
    """Evaluate every raw composite key without a partial/don't-care branch."""

    if not isinstance(key, CompositeKey):
        raise TypeError("key must be a CompositeKey")
    nominal = nominal_cell(key.bank_id, key.discriminator_word)
    transition = transition_cell(
        key.phase_frame,
        key.event_class,
        key.leakage_reset_fsm_state,
        nominal.action_index,
    )
    reason, error_flags = _compose_reason_error(nominal, transition)
    fault = _factorized_fault_id(key, nominal)
    word = _action_from_factors(
        bank=key.bank_id,
        phase=key.phase_frame,
        event=key.event_class,
        state=key.leakage_reset_fsm_state,
        nominal=nominal,
        transition=transition,
    )
    # Pack/unpack is part of evaluation, so CRC and reserved-bit errors cannot
    # remain an untested serializer side path.
    if ActionWord.unpack(word.pack()) != word:
        raise AssertionError("80-bit action word failed exact round trip")
    return RecurrenceResult(
        action_word=word,
        next_phase_frame=transition.next_phase_frame,
        next_fsm_state=transition.next_fsm_state,
        reason_code=reason,
        error_flags=error_flags,
        fault_id=fault,
    )


def fault_response_witnesses() -> tuple[dict[str, Any], ...]:
    """Execute every frozen raw-fault class through canonicalization and F."""

    normal_state = encode_fsm(FsmMode.NORMAL, 0)
    base_key = CompositeKey(
        BankId.A,
        0,
        0,
        EventClass.NORMAL,
        normal_state,
    )

    def failed_integrity(field: str) -> IntegrityStatus:
        values = {
            name: True
            for name in IntegrityStatus.__dataclass_fields__
        }
        values[field] = False
        return IntegrityStatus(**values)

    cases: tuple[
        tuple[
            str,
            tuple[tuple[str, CompositeKey, IntegrityStatus], ...],
        ],
        ...,
    ] = (
        (
            "INVALID_KEY",
            (
                (
                    "invalid_bank",
                    CompositeKey(
                        BankId.INVALID,
                        0,
                        0,
                        EventClass.NORMAL,
                        normal_state,
                    ),
                    IntegrityStatus(),
                ),
                (
                    "invalid_fsm",
                    CompositeKey(
                        BankId.A,
                        0,
                        0,
                        EventClass.NORMAL,
                        255,
                    ),
                    IntegrityStatus(),
                ),
            ),
        ),
        (
            "RESERVED_ACTION_OR_EVENT",
            (
                (
                    "reserved_event",
                    CompositeKey(
                        BankId.A,
                        0,
                        0,
                        EventClass.RESERVED_15,
                        normal_state,
                    ),
                    IntegrityStatus(),
                ),
            ),
        ),
        ("INPUT_CRC", (("input_crc", base_key, failed_integrity("input_crc_ok")),)),
        ("IMAGE_CRC", (("image_crc", base_key, failed_integrity("image_crc_ok")),)),
        ("IMAGE_SHA", (("image_sha", base_key, failed_integrity("image_sha_ok")),)),
        (
            "PARTIAL_PACKAGE",
            (("partial_package", base_key, failed_integrity("package_complete")),),
        ),
        (
            "UNKNOWN_VERSION",
            (("unknown_version", base_key, failed_integrity("version_known")),),
        ),
        (
            "VERSION_MISMATCH",
            (("version_mismatch", base_key, failed_integrity("version_matches")),),
        ),
        (
            "ROLLBACK_VERSION",
            (("rollback_version", base_key, failed_integrity("no_version_rollback")),),
        ),
        (
            "STALE_PACKAGE",
            (("stale_package", base_key, failed_integrity("package_fresh")),),
        ),
        (
            "DEADLINE_MISS",
            (("deadline_miss", base_key, failed_integrity("deadline_met")),),
        ),
        (
            "OOD_WORD",
            (
                (
                    "reserved_discriminator_word",
                    CompositeKey(
                        BankId.A,
                        1,
                        0,
                        EventClass.OOD,
                        normal_state,
                    ),
                    IntegrityStatus(),
                ),
            ),
        ),
        (
            "RESET_ACK_UNEXPECTED",
            (
                (
                    "unexpected_reset_ack",
                    base_key,
                    failed_integrity("reset_ack_expected"),
                ),
            ),
        ),
        (
            "PERSISTENT_LEAKAGE",
            (
                (
                    "third_leakage_observation",
                    CompositeKey(
                        BankId.A,
                        0,
                        0,
                        EventClass.LEAKAGE,
                        encode_fsm(FsmMode.LEAKAGE, 2),
                    ),
                    IntegrityStatus(),
                ),
            ),
        ),
    )
    witnesses: list[dict[str, Any]] = []
    for fault_id, raw_cases in cases:
        results: list[dict[str, Any]] = []
        for case_id, raw_key, integrity in raw_cases:
            canonical = canonicalize_composite_key(raw_key, integrity)
            if canonical.source_fault_id != fault_id:
                raise AssertionError(
                    f"{case_id} classified as "
                    f"{canonical.source_fault_id}, expected {fault_id}"
                )
            recurrence = total_recurrence(canonical.key)
            action = NominalAction(
                recurrence.action_word.action_code
            ).name
            reason = ReasonCode(recurrence.reason_code).name
            if action not in {"LKG_HOLD", "RESET"}:
                raise AssertionError(
                    f"{case_id} did not reach a fail-closed terminal"
                )
            results.append(
                {
                    "case_id": case_id,
                    "raw_key_word": raw_key.to_word(),
                    "canonical_key_word": canonical.key.to_word(),
                    "terminal": action,
                    "reason_code": reason,
                    "fallback": recurrence.action_word.fallback,
                    "hold": recurrence.action_word.hold,
                    "reset_request": (
                        recurrence.action_word.reset_request
                    ),
                }
            )
        reason_codes = list(
            dict.fromkeys(row["reason_code"] for row in results)
        )
        terminals = list(
            dict.fromkeys(row["terminal"] for row in results)
        )
        if len(terminals) != 1:
            raise AssertionError(
                f"{fault_id} has non-unique bounded terminal"
            )
        witnesses.append(
            {
                "fault_id": fault_id,
                "terminal": terminals[0],
                "reason_codes": reason_codes,
                "undefined_action": False,
                "cases": results,
            }
        )
    if tuple(row["fault_id"] for row in witnesses) != FAULT_PRIORITY:
        raise AssertionError("fault witness order differs from priority")
    return tuple(witnesses)


def _fsm_successors(state: int) -> set[int]:
    successors: set[int] = set()
    for event in range(16):
        for nominal in range(8):
            successors.add(
                transition_cell(0, event, state, nominal).next_fsm_state
            )
    return successors


def reachable_fsm_states() -> tuple[int, ...]:
    """Reachability in the complete syntactic T domain, not observed causality."""

    start = encode_fsm(FsmMode.NORMAL, 0)
    visited = {start}
    queue: deque[int] = deque((start,))
    while queue:
        current = queue.popleft()
        for successor in _fsm_successors(current):
            try:
                decode_fsm(successor)
            except ValueError:
                raise AssertionError("transition emitted reserved FSM encoding")
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)
    return tuple(sorted(visited))


def reset_distance(state: int) -> int:
    """Syntactic ACK-success/hysteresis witness, without receipt gating."""

    decode_fsm(state)
    target = encode_fsm(FsmMode.NORMAL, 0)
    if state == target:
        return 0
    current = transition_cell(
        0, EventClass.RESET_ACK_SUCCESS, state, NominalAction.IDLE
    ).next_fsm_state
    for distance in range(1, 6):
        if current == target:
            return distance
        current = transition_cell(
            0, EventClass.NORMAL, current, NominalAction.IDLE
        ).next_fsm_state
    raise AssertionError("valid FSM state lacks bounded reset witness")


def _enumerate_nominal() -> tuple[int, int, str]:
    digest = hashlib.sha256()
    keys: set[int] = set()
    count = 0
    for bank in range(4):
        for word in range(256):
            key_id = (bank << 8) | word
            result = nominal_cell(bank, word)
            digest.update(key_id.to_bytes(2, "big"))
            digest.update(result.to_bytes())
            keys.add(key_id)
            count += 1
    return count, len(keys), digest.hexdigest()


def _enumerate_transition() -> tuple[int, int, str]:
    digest = hashlib.sha256()
    keys: set[int] = set()
    count = 0
    for phase in range(4):
        for event in range(16):
            for state in range(256):
                for nominal in range(8):
                    key_id = (
                        (((phase * 16) + event) * 256 + state) * 8
                        + nominal
                    )
                    result = transition_cell(
                        phase, event, state, nominal
                    )
                    digest.update(key_id.to_bytes(3, "big"))
                    digest.update(result.to_bytes())
                    keys.add(key_id)
                    count += 1
    return count, len(keys), digest.hexdigest()


def _nominal_signatures() -> tuple[tuple[int, int, int, int], ...]:
    signatures = {
        (
            bank,
            int(result.action_index),
            int(result.reason_code),
            int(result.error_flags),
        )
        for bank in range(4)
        for word in range(256)
        for result in (nominal_cell(bank, word),)
    }
    return tuple(sorted(signatures))


def _nominal_equivalence_witness() -> dict[str, Any]:
    """Map every raw N key to its recurrence-equivalence signature."""

    signatures = _nominal_signatures()
    signature_index = {
        signature: index for index, signature in enumerate(signatures)
    }
    class_sizes = [0] * len(signatures)
    representatives: list[dict[str, int] | None] = [
        None
    ] * len(signatures)
    digest = hashlib.sha256()
    mapped_count = 0
    for bank in range(4):
        for word in range(256):
            nominal = nominal_cell(bank, word)
            signature = (
                bank,
                int(nominal.action_index),
                int(nominal.reason_code),
                int(nominal.error_flags),
            )
            index = signature_index[signature]
            raw_key = (bank << 8) | word
            digest.update(raw_key.to_bytes(2, "big"))
            digest.update(index.to_bytes(1, "big"))
            class_sizes[index] += 1
            if representatives[index] is None:
                representatives[index] = {
                    "bank_id": bank,
                    "discriminator_word": word,
                }
            mapped_count += 1
    signature_records = [
        {
            "signature_index": index,
            "bank_id": signature[0],
            "nominal_action_index": signature[1],
            "reason_code": signature[2],
            "error_flags": signature[3],
            "class_size": class_sizes[index],
            "representative_key": representatives[index],
        }
        for index, signature in enumerate(signatures)
    ]
    return {
        "mapped_nominal_key_count": mapped_count,
        "signature_count": len(signatures),
        "signature_class_sizes": class_sizes,
        "signature_class_size_sum": sum(class_sizes),
        "all_signature_classes_nonempty": all(
            size > 0 for size in class_sizes
        ),
        "signatures": signature_records,
        "signatures_sha256": hashlib.sha256(
            canonical_json(signature_records).encode("utf-8")
        ).hexdigest(),
        "raw_key_to_signature_sha256": digest.hexdigest(),
    }


def _action_from_factors(
    *,
    bank: int,
    phase: int,
    event: int,
    state: int,
    nominal: NominalResult,
    transition: TransitionResult,
) -> ActionWord:
    reason, error_flags = _compose_reason_error(nominal, transition)
    return ActionWord(
        action_code=transition.action_code,
        correction_enable=transition.correction_enable,
        reset_request=transition.reset_request,
        fallback=transition.fallback,
        hold=transition.hold,
        pauli_dx=transition.pauli_dx,
        pauli_dz=transition.pauli_dz,
        next_phase_frame=transition.next_phase_frame,
        next_fsm_state=transition.next_fsm_state,
        catalog_action_id=transition.catalog_action_id,
        reason_code=reason,
        error_flags=error_flags,
        source_bank_id=bank,
        factor_tag=_factor_tag(
            bank, phase, event, state, nominal.action_index
        ),
    )


def _enumerate_composition() -> tuple[int, int, int, str]:
    """Enumerate the lossless quotient of all N outputs through T and CRC.

    Discriminator words with identical ``(bank, action, reason, error)`` are
    observationally equivalent to the recurrence.  Enumerating each unique N
    signature across the complete phase/event/FSM domain therefore covers the
    full 16,777,216-key Cartesian domain without sampling.
    """

    signatures = _nominal_signatures()
    digest = hashlib.sha256()
    keys: set[int] = set()
    count = 0
    for signature_index, (bank, action, reason, error) in enumerate(
        signatures
    ):
        nominal = NominalResult(action, reason, error)
        for phase in range(4):
            for event in range(16):
                for state in range(256):
                    transition = transition_cell(
                        phase, event, state, action
                    )
                    word = _action_from_factors(
                        bank=bank,
                        phase=phase,
                        event=event,
                        state=state,
                        nominal=nominal,
                        transition=transition,
                    )
                    quotient_key = (
                        (((signature_index * 4) + phase) * 16 + event)
                        * 256
                        + state
                    )
                    digest.update(quotient_key.to_bytes(3, "big"))
                    digest.update(word.to_bytes())
                    keys.add(quotient_key)
                    count += 1
    return count, len(keys), len(signatures), digest.hexdigest()


def _build_factorized_map_manifest() -> dict[str, Any]:
    """Exhaustively enumerate N and T twice and return immutable witnesses."""

    nominal_first = _enumerate_nominal()
    transition_first = _enumerate_transition()
    composition_first = _enumerate_composition()
    nominal_second = _enumerate_nominal()
    transition_second = _enumerate_transition()
    composition_second = _enumerate_composition()
    equivalence = _nominal_equivalence_witness()
    reachable = reachable_fsm_states()
    valid_states = tuple(
        state
        for state in range(256)
        if (state >> 5) in {int(mode) for mode in FsmMode}
    )
    distances = {state: reset_distance(state) for state in valid_states}
    combined = hashlib.sha256(
        canonical_json(
            {
                "nominal": nominal_first,
                "transition": transition_first,
                "composition": composition_first,
                "layout": ACTION_LAYOUT,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_id": "phase9-factorized-total-recurrence-v1",
        "nominal_expected_count": NOMINAL_CELL_COUNT,
        "nominal_count": nominal_first[0],
        "nominal_unique_keys": nominal_first[1],
        "nominal_sha256": nominal_first[2],
        "transition_expected_count": TRANSITION_CELL_COUNT,
        "transition_count": transition_first[0],
        "transition_unique_keys": transition_first[1],
        "transition_sha256": transition_first[2],
        "composition_expected_count": (
            composition_first[2] * 4 * 16 * 256
        ),
        "composition_count": composition_first[0],
        "composition_unique_keys": composition_first[1],
        "nominal_signature_count": composition_first[2],
        "composition_sha256": composition_first[3],
        "full_cartesian_key_count": (
            NOMINAL_CELL_COUNT * 4 * 16 * 256
        ),
        "composition_quotient_is_lossless": (
            equivalence["mapped_nominal_key_count"]
            == NOMINAL_CELL_COUNT
            == equivalence["signature_class_size_sum"]
            and equivalence["signature_count"] == composition_first[2]
            and equivalence["all_signature_classes_nonempty"]
        ),
        "nominal_equivalence_witness": equivalence,
        "composition_equivalence": (
            "same bank/action/reason/error N signature implies identical "
            "T/action-pack behavior for every phase/event/FSM state"
        ),
        "combined_sha256": combined,
        "deterministic": (
            nominal_first == nominal_second
            and transition_first == transition_second
            and composition_first == composition_second
        ),
        "coverage_complete": (
            nominal_first[0] == NOMINAL_CELL_COUNT
            and transition_first[0] == TRANSITION_CELL_COUNT
            and composition_first[0]
            == composition_first[2] * 4 * 16 * 256
        ),
        "unique_complete": (
            nominal_first[1] == NOMINAL_CELL_COUNT
            and transition_first[1] == TRANSITION_CELL_COUNT
            and composition_first[1] == composition_first[0]
        ),
        "base_residual_zero": True,
        "legal_discriminator_count": len(legal_discriminator_words()),
        "phase_state_count": 4,
        "event_class_count": 16,
        "valid_event_class_count": 13,
        "fsm_state_count": 256,
        "valid_fsm_state_count": len(valid_states),
        "nominal_action_count": 8,
        "reachable_fsm_count": len(reachable),
        "fsm_reachability_scope": (
            "SYNTACTIC_T_DOMAIN_ALL_16_EVENTS_AND_8_NOMINAL_ACTIONS_"
            "NOT_DEPLOYABLE_CAUSAL_REACHABILITY"
        ),
        "reachable_fsm_sha256": hashlib.sha256(
            bytes(reachable)
        ).hexdigest(),
        "reset_bfs_covered_count": len(distances),
        "reset_bfs_max_distance": max(distances.values()),
        "reset_bfs_scope": (
            "SYNTACTIC_SUCCESS_ACK_THEN_HYSTERESIS_"
            "NOT_PREVIOUS_RECEIPT_GATED"
        ),
    }


@lru_cache(maxsize=1)
def _factorized_manifest_json() -> str:
    return canonical_json(_build_factorized_map_manifest())


def factorized_map_manifest() -> dict[str, Any]:
    """Return a fresh copy of the cached exhaustive enumeration witness."""

    return json.loads(_factorized_manifest_json())


@dataclass(frozen=True)
class RepresentativeActionProbe:
    probe_id: str
    intent: str
    input_pattern: str
    expected_terminal: str
    coverage_tags: tuple[str, ...]
    fixture: Mapping[str, Any]
    expected_reason: str
    probe_only: bool = True
    codebook_candidate: bool = False
    performance_evidence: bool = False
    ranking_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "intent": self.intent,
            "input_pattern": self.input_pattern,
            "expected_terminal": self.expected_terminal,
            "coverage_tags": list(self.coverage_tags),
            "fixture": dict(self.fixture),
            "expected_reason": self.expected_reason,
            "probe_only": self.probe_only,
            "codebook_candidate": self.codebook_candidate,
            "performance_evidence": self.performance_evidence,
            "ranking_evidence": self.ranking_evidence,
        }


def representative_action_probes() -> tuple[RepresentativeActionProbe, ...]:
    common = ("pre_codebook_interface_probe",)
    rows = (
        ("P01_IDLE", "neutral idle", "legal centre word", "IDLE", ("idle",), {"steps": [{"discriminator_word": 0}]}, "NOMINAL"),
        ("P02_Q_POS", "positive q correction", "q positive", "X", ("q_positive",), {"steps": [{"discriminator_word": 44}]}, "NOMINAL"),
        ("P03_Q_NEG", "negative q correction", "q negative", "X", ("q_negative",), {"steps": [{"discriminator_word": 18}]}, "NOMINAL"),
        ("P04_P_POS", "positive p correction", "p positive", "Z", ("p_positive",), {"steps": [{"discriminator_word": 172}]}, "NOMINAL"),
        ("P05_P_NEG", "negative p correction", "p negative", "Z", ("p_negative",), {"steps": [{"discriminator_word": 146}]}, "NOMINAL"),
        ("P06_ALTERNATE", "alternating axes", "q,p,q", "X", ("alternating_axes",), {"steps": [{"discriminator_word": 44}, {"discriminator_word": 172}, {"discriminator_word": 44}]}, "NOMINAL"),
        ("P07_BOUNDARY", "quantizer boundary", "largest legal class", "XZ", ("boundary",), {"steps": [{"discriminator_word": 252}]}, "NOMINAL"),
        ("P08_PHASE", "phase-frame recurrence", "nonzero phase", "X", ("phase_frame",), {"memory": {"phase_frame": 3}, "steps": [{"discriminator_word": 44}]}, "NOMINAL"),
        ("P09_LEAK_RESET", "persistent leakage", "three leakage events", "RESET", ("leakage", "persistent_leakage"), {"steps": [{"leakage_confidence": 255}, {"leakage_confidence": 255}, {"leakage_confidence": 255}]}, "PERSISTENT_LEAKAGE"),
        ("P10_RESET_OK", "observed reset success", "valid prior reset receipt", "HOLD", ("reset_success",), {"previous_reset_receipt": True, "steps": [{"reset_ack": "success"}]}, "RESET_SUCCESS"),
        ("P11_RESET_FAIL", "observed reset failure", "valid prior reset receipt", "RESET", ("reset_failure",), {"previous_reset_receipt": True, "steps": [{"reset_ack": "failure"}]}, "RESET_FAILURE"),
        ("P12_BAD_CRC", "integrity rejection", "input CRC fault bit", "LKG_HOLD", ("crc_version_stale_deadline", "integrity_fault"), {"memory": {"integrity_flags": 1 << INTEGRITY_FLAG_LAYOUT["INPUT_CRC"]}, "steps": [{}]}, "INTEGRITY_FAULT"),
        ("P13_STALE", "stale/version rejection", "stale package fault bit", "LKG_HOLD", ("crc_version_stale_deadline", "version_fault"), {"memory": {"integrity_flags": 1 << INTEGRITY_FLAG_LAYOUT["STALE_PACKAGE"]}, "steps": [{}]}, "VERSION_FAULT"),
        ("P14_OOD", "low-confidence hold then OOD abstention", "low confidence followed by reserved discriminator word", "LKG_HOLD", ("ood", "low_confidence"), {"steps": [{"discriminator_confidence": 0}, {"discriminator_word": 1}]}, "OOD_WORD"),
        ("P15_DEADLINE", "transport/deadline rejection", "deadline miss bit", "LKG_HOLD", ("crc_version_stale_deadline", "transport_fault"), {"memory": {"integrity_flags": 1 << INTEGRITY_FLAG_LAYOUT["DEADLINE_MISS"]}, "steps": [{}]}, "DEADLINE_FAULT"),
        ("P16_LKG_RECOVERY", "LKG trusted-bank hold", "trusted bank handle", "LKG_HOLD", ("lkg_recovery",), {"memory": {"bank_id": int(BankId.LKG)}, "steps": [{}]}, "LKG_ACTIVE"),
    )
    return tuple(
        RepresentativeActionProbe(
            probe_id,
            intent,
            pattern,
            terminal,
            tuple(tags) + common,
            fixture,
            expected_reason,
        )
        for (
            probe_id,
            intent,
            pattern,
            terminal,
            tags,
            fixture,
            expected_reason,
        ) in rows
    )


def validate_complete_package_nomination(
    package: Mapping[str, Any],
    *,
    current_version: int,
    minimum_safe_activation_epoch: int,
    trusted_schema_ids: Sequence[str],
    trusted_package_ids: Sequence[str],
    trusted_provenance_sha256: Sequence[str],
    expected_release_pin_sha256: str,
) -> None:
    required = {
        "schema_id",
        "package_id",
        "bank_target",
        "version",
        "activation_epoch",
        "word_count",
        "crc32",
        "content_sha256",
        "provenance_sha256",
        "release_pin_sha256",
        "entries",
    }
    if set(package) != required:
        raise ValueError("slow path must nominate exactly one complete package")
    version = _exact_uint(package["version"], 16, "version")
    current = _exact_uint(current_version, 16, "current_version")
    if version <= current:
        raise ValueError(
            "LKG/updated package must use a strictly higher version"
        )
    activation_epoch = _exact_uint(
        package["activation_epoch"], 64, "activation_epoch"
    )
    safe_epoch = _exact_uint(
        minimum_safe_activation_epoch,
        64,
        "minimum_safe_activation_epoch",
    )
    if activation_epoch < safe_epoch:
        raise ValueError("activation epoch precedes the safe boundary")
    if package["schema_id"] not in set(trusted_schema_ids):
        raise ValueError("schema_id is not in the trusted registry")
    if package["package_id"] not in set(trusted_package_ids):
        raise ValueError("package_id is not in the trusted nomination ledger")
    if package["bank_target"] not in {"A", "B"}:
        raise ValueError("bank_target must be inactive physical bank A or B")
    entries = package["entries"]
    if not isinstance(entries, (list, tuple)) or len(entries) != 256:
        raise ValueError("package must contain exactly 256 precompiled entries")
    if package["word_count"] != 256:
        raise ValueError("word_count must exactly match complete package")
    for entry in entries:
        _exact_uint(entry, 64, "package entry")
    content = b"".join(int(entry).to_bytes(8, "little") for entry in entries)
    if package["crc32"] != zlib.crc32(content) & 0xFFFFFFFF:
        raise ValueError("package CRC32 mismatch")
    if package["content_sha256"] != hashlib.sha256(content).hexdigest():
        raise ValueError("package content SHA-256 mismatch")
    provenance = package["provenance_sha256"]
    if not (
        isinstance(provenance, str)
        and len(provenance) == 64
        and all(char in "0123456789abcdef" for char in provenance)
        and provenance != "0" * 64
        and provenance in set(trusted_provenance_sha256)
    ):
        raise ValueError("package provenance is not trusted")
    release_pin = package["release_pin_sha256"]
    if not (
        isinstance(expected_release_pin_sha256, str)
        and len(expected_release_pin_sha256) == 64
        and all(
            char in "0123456789abcdef"
            for char in expected_release_pin_sha256
        )
        and expected_release_pin_sha256 != "0" * 64
        and release_pin == expected_release_pin_sha256
    ):
        raise ValueError("package release-pin binding mismatch")


def _minimal_observed(word: int = 0) -> dict[str, Any]:
    return {
        "iq_i": [0, 1],
        "iq_q": [0, -1],
        "iq_source": "synthetic",
        "matched_filter_i": 0,
        "matched_filter_q": 0,
        "llr_q": 0,
        "llr_p": 0,
        "discriminator_word": word,
        "discriminator_confidence": 255,
        "leakage_confidence": 0,
        "timestamp": 0,
        "reset_ack": "none",
        "previous_action_word": 0,
        "previous_composite_key": 0,
        "previous_action_present": False,
        "previous_active_image_version": 0,
    }


def _minimal_memory() -> dict[str, Any]:
    return {
        "bank_id": 0,
        "image_version": 1,
        "trusted_version": 1,
        "phase_frame": 0,
        "pauli_frame": 0,
        "previous_event_class": 0,
        "leakage_reset_fsm_state": encode_fsm(FsmMode.NORMAL, 0),
        "integrity_flags": 0,
    }


@dataclass(frozen=True)
class DecisionReceipt:
    assembly: KeyAssembly
    recurrence: RecurrenceResult
    active_image_version_sideband: int


def deployable_decision(
    observed: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> DecisionReceipt:
    """Run the sole deployable one-round path; no future/truth argument exists."""

    assembly = assemble_deployable_key(observed, memory)
    recurrence = total_recurrence(assembly.key)
    return DecisionReceipt(
        assembly=assembly,
        recurrence=recurrence,
        active_image_version_sideband=memory["image_version"],
    )


def causal_prefix_token(
    observed: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> str:
    decision = deployable_decision(observed, memory)
    return hashlib.sha256(
        decision.recurrence.action_word.to_bytes()
        + decision.active_image_version_sideband.to_bytes(2, "little")
    ).hexdigest()


def execute_representative_probe(
    probe: RepresentativeActionProbe,
) -> tuple[DecisionReceipt, ...]:
    if not isinstance(probe, RepresentativeActionProbe):
        raise TypeError("probe must be a RepresentativeActionProbe")
    memory = _minimal_memory()
    memory.update(dict(probe.fixture.get("memory", {})))
    prior_word = 0
    prior_key_word = 0
    prior_present = False
    prior_version = 0
    timestamp_offset = 0
    if probe.fixture.get("previous_reset_receipt") is True:
        # Produce the prior RESET through the actual observed-only deployable
        # path.  Three leakage observations are the shortest reachable trace
        # from NORMAL to a reset-request receipt; no hand-built receipt is
        # accepted as evidence for the reset-ack probes.
        for prelude_timestamp in range(3):
            prelude = _minimal_observed()
            prelude["timestamp"] = prelude_timestamp
            prelude["leakage_confidence"] = 255
            prelude["previous_action_word"] = prior_word
            prelude["previous_composite_key"] = prior_key_word
            prelude["previous_action_present"] = prior_present
            prelude["previous_active_image_version"] = prior_version
            prior_receipt = deployable_decision(prelude, memory)
            memory = dict(memory)
            memory["phase_frame"] = (
                prior_receipt.recurrence.next_phase_frame
            )
            memory["leakage_reset_fsm_state"] = (
                prior_receipt.recurrence.next_fsm_state
            )
            memory["previous_event_class"] = (
                prior_receipt.assembly.key.event_class
            )
            memory["integrity_flags"] = 0
            prior_word = prior_receipt.recurrence.action_word.pack()
            prior_key_word = prior_receipt.assembly.key.to_word()
            prior_present = True
            prior_version = (
                prior_receipt.active_image_version_sideband
            )
        if not prior_receipt.recurrence.action_word.reset_request:
            raise AssertionError(
                "reachable reset prelude did not produce a reset request"
            )
        timestamp_offset = 3
    receipts: list[DecisionReceipt] = []
    steps = probe.fixture.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("probe fixture requires a nonempty step list")
    for timestamp, overrides in enumerate(steps):
        observed = _minimal_observed()
        observed.update(dict(overrides))
        observed["timestamp"] = timestamp_offset + timestamp
        observed["previous_action_word"] = prior_word
        observed["previous_composite_key"] = prior_key_word
        observed["previous_action_present"] = prior_present
        observed["previous_active_image_version"] = prior_version
        receipt = deployable_decision(observed, memory)
        receipts.append(receipt)
        memory = dict(memory)
        memory["phase_frame"] = receipt.recurrence.next_phase_frame
        memory["leakage_reset_fsm_state"] = (
            receipt.recurrence.next_fsm_state
        )
        memory["previous_event_class"] = receipt.assembly.key.event_class
        memory["integrity_flags"] = 0
        prior_word = receipt.recurrence.action_word.pack()
        prior_key_word = receipt.assembly.key.to_word()
        prior_present = True
        prior_version = receipt.active_image_version_sideband
    final = receipts[-1].recurrence
    if NominalAction(final.action_word.action_code).name != (
        probe.expected_terminal
    ):
        raise AssertionError(
            f"{probe.probe_id} terminal mismatch: "
            f"{NominalAction(final.action_word.action_code).name}"
        )
    if ReasonCode(final.reason_code).name != probe.expected_reason:
        raise AssertionError(
            f"{probe.probe_id} reason mismatch: "
            f"{ReasonCode(final.reason_code).name}"
        )
    witnessed = probe_coverage_witnesses(receipts)
    required = set(probe.coverage_tags) - {
        "pre_codebook_interface_probe"
    }
    missing = required - witnessed
    if missing:
        raise AssertionError(
            f"{probe.probe_id} coverage tags lack runtime witnesses: "
            f"{sorted(missing)}"
        )
    return tuple(receipts)


def probe_coverage_witnesses(
    receipts: Sequence[DecisionReceipt],
) -> set[str]:
    """Derive probe coverage labels only from executed recurrence receipts."""

    if not receipts:
        return set()
    events = [
        EventClass(receipt.assembly.key.event_class)
        for receipt in receipts
    ]
    actions = [
        NominalAction(receipt.recurrence.action_word.action_code)
        for receipt in receipts
    ]
    reasons = [
        ReasonCode(receipt.recurrence.reason_code)
        for receipt in receipts
    ]
    witnessed: set[str] = set()
    if actions[-1] == NominalAction.IDLE:
        witnessed.add("idle")
    event_tags = {
        EventClass.Q_POSITIVE: "q_positive",
        EventClass.Q_NEGATIVE: "q_negative",
        EventClass.P_POSITIVE: "p_positive",
        EventClass.P_NEGATIVE: "p_negative",
    }
    witnessed.update(
        event_tags[event] for event in events if event in event_tags
    )
    if actions[-1] == NominalAction.XZ:
        witnessed.add("boundary")
    if any(receipt.assembly.key.phase_frame != 0 for receipt in receipts):
        witnessed.add("phase_frame")
    if len(events) >= 3 and events[-3:] == [
        EventClass.Q_POSITIVE,
        EventClass.P_POSITIVE,
        EventClass.Q_POSITIVE,
    ] and actions[-3:] == [
        NominalAction.X,
        NominalAction.Z,
        NominalAction.X,
    ]:
        witnessed.add("alternating_axes")
    if EventClass.LEAKAGE in events:
        witnessed.add("leakage")
    reason_tags = {
        ReasonCode.PERSISTENT_LEAKAGE: "persistent_leakage",
        ReasonCode.RESET_SUCCESS: "reset_success",
        ReasonCode.RESET_FAILURE: "reset_failure",
        ReasonCode.OOD_WORD: "ood",
        ReasonCode.LOW_CONFIDENCE: "low_confidence",
        ReasonCode.INTEGRITY_FAULT: "integrity_fault",
        ReasonCode.VERSION_FAULT: "version_fault",
        ReasonCode.DEADLINE_FAULT: "transport_fault",
        ReasonCode.LKG_ACTIVE: "lkg_recovery",
    }
    witnessed.update(
        reason_tags[reason] for reason in reasons if reason in reason_tags
    )
    if any(
        reason
        in {
            ReasonCode.INTEGRITY_FAULT,
            ReasonCode.VERSION_FAULT,
            ReasonCode.DEADLINE_FAULT,
        }
        for reason in reasons
    ):
        witnessed.add("crc_version_stale_deadline")
    return witnessed


def audit_contract() -> dict[str, Any]:
    manifest = factorized_map_manifest()
    observed = _minimal_observed()
    memory = _minimal_memory()
    validate_deployable_inputs(observed, memory)
    key = CompositeKey(0, 0, 0, 0, encode_fsm(FsmMode.NORMAL, 0))
    nominal_result = total_recurrence(key)
    invalid_result = total_recurrence(
        CompositeKey(3, 255, 0, 15, 255)
    )
    decision = deployable_decision(observed, memory)
    ood_observed = dict(observed)
    ood_observed["discriminator_word"] = 1
    ood_decision = deployable_decision(ood_observed, memory)
    leak_receipts = execute_representative_probe(
        representative_action_probes()[8]
    )
    probe_receipts = [
        execute_representative_probe(probe)
        for probe in representative_action_probes()
    ]
    fault_witnesses = fault_response_witnesses()
    package_entries = list(range(256))
    package_content = b"".join(
        entry.to_bytes(8, "little") for entry in package_entries
    )
    package_provenance = hashlib.sha256(b"trusted-provenance").hexdigest()
    package_release_pin = hashlib.sha256(b"external-release-pin").hexdigest()
    sample_package = {
        "schema_id": "phase9-probe-package-v1",
        "package_id": "probe-package-001",
        "bank_target": "B",
        "version": 2,
        "activation_epoch": 10,
        "word_count": 256,
        "crc32": zlib.crc32(package_content) & 0xFFFFFFFF,
        "content_sha256": hashlib.sha256(package_content).hexdigest(),
        "provenance_sha256": package_provenance,
        "release_pin_sha256": package_release_pin,
        "entries": package_entries,
    }
    validate_complete_package_nomination(
        sample_package,
        current_version=1,
        minimum_safe_activation_epoch=10,
        trusted_schema_ids=["phase9-probe-package-v1"],
        trusted_package_ids=["probe-package-001"],
        trusted_provenance_sha256=[package_provenance],
        expected_release_pin_sha256=package_release_pin,
    )
    checks = {
        "exact_namespace_ids": tuple(NAMESPACE_SCHEMAS)
        == (
            "BACKEND_LATENT",
            "DEPLOYABLE_OBSERVED",
            "CONTROLLER_MEMORY",
            "EVALUATOR_TRUTH",
            "PROVENANCE",
        ),
        "deployable_namespaces_only_observed_and_memory": [
            name
            for name, schema in NAMESPACE_SCHEMAS.items()
            if schema["deployable"]
        ]
        == ["DEPLOYABLE_OBSERVED", "CONTROLLER_MEMORY"],
        "legal_discriminator_count_126": len(legal_discriminator_words())
        == 126,
        "action_layout_exact_80": sum(
            field["bits"] for field in ACTION_LAYOUT
        )
        == ACTION_WORD_BITS,
        "phase_semantics_exact_four_states": set(
            PHASE_FRAME_SEMANTICS["states"]
        )
        == {"0", "1", "2", "3"}
        and PHASE_FRAME_SEMANTICS[
            "current_rtl_two_uint8_adapter_qualified"
        ]
        is False,
        "fsm_is_new_unqualified_adapter_contract": FSM_ENCODING[
            "current_rtl_six_counter_adapter_qualified"
        ]
        is False,
        "nominal_total_1024": manifest["nominal_count"]
        == NOMINAL_CELL_COUNT
        == manifest["nominal_unique_keys"],
        "transition_total_131072": manifest["transition_count"]
        == TRANSITION_CELL_COUNT
        == manifest["transition_unique_keys"],
        "composition_quotient_is_lossless_and_crc_enumerated": (
            manifest["composition_count"]
            == manifest["composition_expected_count"]
            == manifest["composition_unique_keys"]
            and manifest["composition_quotient_is_lossless"] is True
            and manifest["full_cartesian_key_count"]
            == NOMINAL_CELL_COUNT * 4 * 16 * 256
            and manifest["nominal_equivalence_witness"][
                "mapped_nominal_key_count"
            ]
            == NOMINAL_CELL_COUNT
            and manifest["nominal_equivalence_witness"][
                "signature_class_size_sum"
            ]
            == NOMINAL_CELL_COUNT
        ),
        "repeat_enumeration_stable": manifest["deterministic"],
        "all_valid_fsm_have_reset_witness": manifest[
            "reset_bfs_covered_count"
        ]
        == 192,
        "future_truth_structurally_absent_from_decision_api": (
            tuple(inspect.signature(deployable_decision).parameters)
            == ("observed", "memory")
            and tuple(inspect.signature(total_recurrence).parameters)
            == ("key",)
        ),
        "assembler_is_live_decision_path": (
            decision.assembly.key
            == assemble_deployable_key(observed, memory).key
            and decision.recurrence
            == total_recurrence(decision.assembly.key)
        ),
        "same_key_same_packed_action": (
            total_recurrence(key).action_word.pack()
            == total_recurrence(key).action_word.pack()
        ),
        "fault_priority_exact": FAULT_PRIORITY
        == (
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
        ),
        "all_fault_responses_are_executable_and_bounded": (
            tuple(row["fault_id"] for row in fault_witnesses)
            == FAULT_PRIORITY
            and all(
                row["terminal"] in {"LKG_HOLD", "RESET"}
                and row["undefined_action"] is False
                and row["cases"]
                and all(
                    case["fallback"]
                    and (
                        case["hold"] or case["reset_request"]
                    )
                    for case in row["cases"]
                )
                for row in fault_witnesses
            )
        ),
        "ood_deployable_path_reachable": (
            ood_decision.recurrence.reason_code == ReasonCode.OOD_WORD
            and ood_decision.recurrence.action_word.action_code
            == NominalAction.LKG_HOLD
        ),
        "persistent_leakage_deployable_path_reachable": (
            len(leak_receipts) == 3
            and leak_receipts[-1].recurrence.action_word.reset_request
            and leak_receipts[-1].recurrence.reason_code
            == ReasonCode.PERSISTENT_LEAKAGE
        ),
        "nominal_roundtrip_crc": ActionWord.unpack(
            nominal_result.action_word.pack()
        )
        == nominal_result.action_word,
        "invalid_key_fail_closed": invalid_result.action_word.fallback
        and (
            invalid_result.action_word.hold
            or invalid_result.action_word.reset_request
        ),
        "base_residual_zero": manifest["base_residual_zero"]
        and nominal_result.action_word.residual_q == 0
        and nominal_result.action_word.residual_p == 0,
        "representative_probe_count_16": len(
            representative_action_probes()
        )
        == 16,
        "all_representative_probes_execute": len(probe_receipts) == 16
        and all(receipts for receipts in probe_receipts),
        "probes_are_nonpromotional": all(
            probe.probe_only
            and not probe.codebook_candidate
            and not probe.performance_evidence
            and not probe.ranking_evidence
            for probe in representative_action_probes()
        ),
        "package_nomination_is_registry_pin_and_epoch_bound": (
            SLOW_PATH_BOUNDARY["requires_trusted_schema_registry"]
            and SLOW_PATH_BOUNDARY["requires_trusted_package_ledger"]
            and SLOW_PATH_BOUNDARY["requires_external_release_pin"]
            and SLOW_PATH_BOUNDARY["atomic_bank_integration_qualified"]
            is False
        ),
        "iq_physical_envelope_is_explicitly_deferred": (
            OBSERVATION_ENVELOPE_BOUNDARY["sample_rate"] is None
            and OBSERVATION_ENVELOPE_BOUNDARY["deferred_to"] == "T9.2.6"
        ),
        "version_is_nondecision_sideband": (
            ACTION_SIDEBAND_CONTRACT[
                "active_image_version_is_decision_input"
            ]
            is False
            and ACTION_SIDEBAND_CONTRACT[
                "current_118bit_rtl_adapter_qualified"
            ]
            is False
        ),
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "manifest_sha256": hashlib.sha256(
            canonical_json(manifest).encode("utf-8")
        ).hexdigest(),
    }


__all__ = [
    "ACTION_LAYOUT",
    "ACTION_SIDEBAND_CONTRACT",
    "ACTION_WORD_BITS",
    "ActionWord",
    "BankId",
    "CLAIM_BOUNDARY",
    "CompositeKey",
    "CRC16_CONTRACT",
    "DISCRIMINATOR_LAYOUT",
    "EventClass",
    "FAULT_PRIORITY",
    "FSM_ENCODING",
    "FsmMode",
    "IntegrityStatus",
    "INTEGRITY_FLAG_LAYOUT",
    "MODEL_SCOPE",
    "NAMESPACE_SCHEMAS",
    "NOMINAL_CELL_COUNT",
    "NominalAction",
    "OBSERVATION_ENVELOPE_BOUNDARY",
    "ReasonCode",
    "PHASE_FRAME_SEMANTICS",
    "SLOW_PATH_BOUNDARY",
    "TRANSITION_CELL_COUNT",
    "TRUTH_PROVENANCE_DENYLIST",
    "assemble_deployable_key",
    "audit_contract",
    "canonicalize_composite_key",
    "causal_prefix_token",
    "classify_raw_fault",
    "crc16_ccitt",
    "decode_fsm",
    "deployable_decision",
    "encode_fsm",
    "execute_representative_probe",
    "factorized_map_manifest",
    "fault_response_witnesses",
    "is_legal_discriminator_word",
    "legal_discriminator_words",
    "nominal_cell",
    "probe_coverage_witnesses",
    "reachable_fsm_states",
    "representative_action_probes",
    "reset_distance",
    "total_recurrence",
    "transition_cell",
    "validate_complete_package_nomination",
    "validate_deployable_inputs",
    "validate_namespace",
]
