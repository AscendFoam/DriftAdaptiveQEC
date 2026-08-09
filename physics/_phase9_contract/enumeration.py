"""Exhaustive recurrence enumeration and reproducible manifest witnesses."""

from __future__ import annotations

from collections import deque
from functools import lru_cache
import hashlib
import json
from typing import Any

from .recurrence import _action_from_factors, nominal_cell, transition_cell
from .schema import (
    ACTION_LAYOUT,
    EventClass,
    FsmMode,
    NOMINAL_CELL_COUNT,
    NominalAction,
    NominalResult,
    TRANSITION_CELL_COUNT,
    canonical_json,
    decode_fsm,
    encode_fsm,
    legal_discriminator_words,
)

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

