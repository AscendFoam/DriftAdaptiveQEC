"""Total recurrence and fault canonicalization for the Phase-9 contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schema import (
    ActionWord,
    BankId,
    CompositeKey,
    EventClass,
    FAULT_PRIORITY,
    FsmMode,
    INTEGRITY_FLAG_LAYOUT,
    IntegrityStatus,
    NominalAction,
    NominalResult,
    OBSERVATION_ENVELOPE_BOUNDARY,
    ReasonCode,
    RecurrenceResult,
    TransitionResult,
    _contains_denied_token,
    _exact_uint,
    _nominal_pauli,
    crc16_ccitt,
    decode_fsm,
    encode_fsm,
    is_legal_discriminator_word,
    validate_namespace,
)

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

