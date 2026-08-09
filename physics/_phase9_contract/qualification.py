"""Package nomination, representative probes, decisions, and contract audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from typing import Any, Mapping, Sequence
import zlib

from .enumeration import factorized_map_manifest
from .recurrence import (
    KeyAssembly,
    assemble_deployable_key,
    fault_response_witnesses,
    total_recurrence,
    validate_deployable_inputs,
)
from .schema import (
    ACTION_LAYOUT,
    ACTION_SIDEBAND_CONTRACT,
    ACTION_WORD_BITS,
    ActionWord,
    BankId,
    CompositeKey,
    EventClass,
    FAULT_PRIORITY,
    FSM_ENCODING,
    FsmMode,
    INTEGRITY_FLAG_LAYOUT,
    NAMESPACE_SCHEMAS,
    NOMINAL_CELL_COUNT,
    NominalAction,
    OBSERVATION_ENVELOPE_BOUNDARY,
    PHASE_FRAME_SEMANTICS,
    ReasonCode,
    RecurrenceResult,
    SLOW_PATH_BOUNDARY,
    TRANSITION_CELL_COUNT,
    _exact_uint,
    canonical_json,
    encode_fsm,
    legal_discriminator_words,
)

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

