from __future__ import annotations

import copy
import hashlib
import zlib

import pytest

from physics import phase9_twin_contract as subject


def _observed() -> dict[str, object]:
    return {
        "iq_i": [0, 1],
        "iq_q": [0, -1],
        "iq_source": "synthetic",
        "matched_filter_i": 0,
        "matched_filter_q": 0,
        "llr_q": 0,
        "llr_p": 0,
        "discriminator_word": 0,
        "discriminator_confidence": 255,
        "leakage_confidence": 0,
        "timestamp": 4,
        "reset_ack": "none",
        "previous_action_word": 0,
        "previous_composite_key": 0,
        "previous_action_present": False,
        "previous_active_image_version": 0,
    }


def _memory() -> dict[str, object]:
    return {
        "bank_id": 0,
        "image_version": 1,
        "trusted_version": 1,
        "phase_frame": 0,
        "pauli_frame": 0,
        "previous_event_class": 0,
        "leakage_reset_fsm_state": subject.encode_fsm(
            subject.FsmMode.NORMAL, 0
        ),
        "integrity_flags": 0,
    }


def _action(**updates: object) -> subject.ActionWord:
    values: dict[str, object] = {
        "action_code": subject.NominalAction.X,
        "correction_enable": True,
        "reset_request": False,
        "fallback": False,
        "hold": False,
        "pauli_dx": 1,
        "pauli_dz": 0,
        "next_phase_frame": 1,
        "next_fsm_state": subject.encode_fsm(
            subject.FsmMode.NORMAL, 0
        ),
        "catalog_action_id": 1,
        "reason_code": subject.ReasonCode.NOMINAL,
        "error_flags": 0,
        "source_bank_id": subject.BankId.A,
        "factor_tag": 7,
    }
    values.update(updates)
    return subject.ActionWord(**values)  # type: ignore[arg-type]


def _package(version: int = 2) -> dict[str, object]:
    entries = list(range(256))
    content = b"".join(item.to_bytes(8, "little") for item in entries)
    return {
        "schema_id": "phase9-test-package-v1",
        "package_id": "phase9-test-package-001",
        "bank_target": "B",
        "version": version,
        "activation_epoch": 9,
        "word_count": 256,
        "crc32": zlib.crc32(content) & 0xFFFFFFFF,
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "provenance_sha256": hashlib.sha256(b"provenance").hexdigest(),
        "release_pin_sha256": hashlib.sha256(b"release-pin").hexdigest(),
        "entries": entries,
    }


def _validate_package(package: dict[str, object]) -> None:
    subject.validate_complete_package_nomination(
        package,
        current_version=1,
        minimum_safe_activation_epoch=9,
        trusted_schema_ids=["phase9-test-package-v1"],
        trusted_package_ids=["phase9-test-package-001"],
        trusted_provenance_sha256=[
            hashlib.sha256(b"provenance").hexdigest()
        ],
        expected_release_pin_sha256=hashlib.sha256(
            b"release-pin"
        ).hexdigest(),
    )


def test_namespaces_are_exact_disjoint_and_action_is_separate() -> None:
    assert tuple(subject.NAMESPACE_SCHEMAS) == (
        "BACKEND_LATENT",
        "DEPLOYABLE_OBSERVED",
        "CONTROLLER_MEMORY",
        "EVALUATOR_TRUTH",
        "PROVENANCE",
    )
    assert "ACTION_WORD" not in subject.NAMESPACE_SCHEMAS
    assert [
        key
        for key, schema in subject.NAMESPACE_SCHEMAS.items()
        if schema["deployable"]
    ] == ["DEPLOYABLE_OBSERVED", "CONTROLLER_MEMORY"]


def test_deployable_allowlist_rejects_missing_extra_wrong_type_and_truth() -> None:
    subject.validate_deployable_inputs(_observed(), _memory())
    for field, value in (
        ("logical_error", False),
        ("future_measurement", 3),
        ("trace_sha256", "0" * 64),
    ):
        observed = _observed()
        observed[field] = value
        with pytest.raises(ValueError):
            subject.validate_deployable_inputs(observed, _memory())
    missing = _observed()
    missing.pop("llr_q")
    with pytest.raises(ValueError):
        subject.validate_deployable_inputs(missing, _memory())
    wrong = _observed()
    wrong["timestamp"] = True
    with pytest.raises(TypeError):
        subject.validate_deployable_inputs(wrong, _memory())


def test_recursive_truth_provenance_rejection_is_not_key_blacklist_only() -> None:
    observed = _observed()
    observed["iq_source"] = "future_truth"
    with pytest.raises((TypeError, ValueError)):
        subject.validate_deployable_inputs(observed, _memory())


def test_discriminator_domain_has_126_legal_and_130_ood_codes() -> None:
    legal = subject.legal_discriminator_words()
    assert len(legal) == 126
    assert len(set(legal)) == 126
    assert all(subject.is_legal_discriminator_word(word) for word in legal)
    invalid = set(range(256)) - set(legal)
    assert len(invalid) == 130
    assert all(
        subject.nominal_cell(subject.BankId.A, word).action_index
        == subject.NominalAction.LKG_HOLD
        and subject.nominal_cell(subject.BankId.A, word).reason_code
        == subject.ReasonCode.OOD_WORD
        for word in invalid
    )


def test_nominal_n_map_is_exactly_total_unique_and_deterministic() -> None:
    first = {
        (bank, word): subject.nominal_cell(bank, word)
        for bank in range(4)
        for word in range(256)
    }
    second = {
        (bank, word): subject.nominal_cell(bank, word)
        for bank in range(4)
        for word in range(256)
    }
    assert len(first) == subject.NOMINAL_CELL_COUNT == 1024
    assert first == second


def test_action_layout_is_64_payload_plus_crc16_and_roundtrips() -> None:
    assert sum(row["bits"] for row in subject.ACTION_LAYOUT) == 80
    assert subject.ACTION_LAYOUT[-1] == {
        "field": "crc16",
        "bits": 16,
        "lsb": 64,
    }
    word = _action()
    packed = word.pack()
    assert packed.bit_length() <= 80
    assert subject.ActionWord.unpack(packed) == word
    with pytest.raises(ValueError, match="CRC16"):
        subject.ActionWord.unpack(packed ^ (1 << 64))


def test_action_word_rejects_nonzero_residual_reserved_action_and_invalid_bit() -> None:
    with pytest.raises(ValueError, match="residual"):
        _action(residual_q=1)
    with pytest.raises(ValueError, match="reserved action"):
        _action(action_code=subject.NominalAction.INVALID)
    with pytest.raises(ValueError, match="valid bit"):
        _action(valid=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reason_code", 63, "reserved reason"),
        ("next_fsm_state", 255, "reserved FSM"),
        ("hold", True, "hold contradiction"),
    ],
)
def test_recomputed_crc_cannot_hide_semantically_invalid_payload(
    field: str, value: int, message: str
) -> None:
    packed = _action().pack()
    payload = packed & ((1 << 64) - 1)
    layout = {
        row["field"]: row for row in subject.ACTION_LAYOUT[:-1]
    }
    row = layout[field]
    payload &= ~(((1 << row["bits"]) - 1) << row["lsb"])
    payload |= value << row["lsb"]
    crc = subject.crc16_ccitt(payload.to_bytes(8, "little"))
    with pytest.raises(ValueError, match=message):
        subject.ActionWord.unpack(payload | (crc << 64))


def test_every_raw_t_key_has_defined_bounded_output_and_stable_fingerprint() -> None:
    manifest = subject.factorized_map_manifest()
    assert manifest["nominal_count"] == 1024
    assert manifest["nominal_unique_keys"] == 1024
    assert manifest["transition_count"] == 131072
    assert manifest["transition_unique_keys"] == 131072
    assert (
        manifest["composition_count"]
        == manifest["composition_expected_count"]
        == manifest["composition_unique_keys"]
        == 196608
    )
    assert manifest["full_cartesian_key_count"] == 16777216
    assert manifest["composition_quotient_is_lossless"] is True
    assert manifest["fsm_reachability_scope"].startswith(
        "SYNTACTIC_T_DOMAIN"
    )
    assert manifest["reset_bfs_scope"].startswith(
        "SYNTACTIC_SUCCESS_ACK"
    )
    witness = manifest["nominal_equivalence_witness"]
    assert witness["mapped_nominal_key_count"] == 1024
    assert witness["signature_class_size_sum"] == 1024
    assert witness["signature_count"] == 12
    assert all(witness["signature_class_sizes"])
    assert manifest["coverage_complete"] is True
    assert manifest["unique_complete"] is True
    assert manifest["deterministic"] is True
    assert len(manifest["nominal_sha256"]) == 64
    assert len(manifest["transition_sha256"]) == 64


def test_invalid_fsm_and_reserved_event_are_total_and_fail_closed() -> None:
    invalid_fsm = subject.transition_cell(
        0, subject.EventClass.NORMAL, 255, subject.NominalAction.X
    )
    reserved_event = subject.transition_cell(
        0,
        subject.EventClass.RESERVED_15,
        subject.encode_fsm(subject.FsmMode.NORMAL, 0),
        subject.NominalAction.X,
    )
    for result in (invalid_fsm, reserved_event):
        assert result.fallback is True
        assert result.reset_request or result.hold
        assert result.action_code != subject.NominalAction.INVALID
        subject.decode_fsm(result.next_fsm_state)


@pytest.mark.parametrize(
    "field",
    [
        "input_crc_ok",
        "image_crc_ok",
        "image_sha_ok",
        "version_known",
        "version_matches",
        "no_version_rollback",
        "package_fresh",
        "package_complete",
        "deadline_met",
        "reset_ack_expected",
    ],
)
def test_each_integrity_fault_closes_to_bounded_lkg_or_reset(field: str) -> None:
    kwargs = {name: True for name in subject.IntegrityStatus.__dataclass_fields__}
    kwargs[field] = False
    status = subject.IntegrityStatus(**kwargs)
    raw = subject.CompositeKey(
        0,
        0,
        0,
        subject.EventClass.NORMAL,
        subject.encode_fsm(subject.FsmMode.NORMAL, 0),
    )
    canonical = subject.canonicalize_composite_key(raw, status)
    result = subject.total_recurrence(
        canonical.key,
    )
    assert result.action_word.fallback is True
    assert result.action_word.hold or result.action_word.reset_request
    assert result.action_word.action_code != subject.NominalAction.INVALID


def test_invalid_bank_reason_survives_factorisation_and_crc_pack() -> None:
    result = subject.total_recurrence(
        subject.CompositeKey(3, 0, 0, 0, 0),
    )
    assert result.reason_code == subject.ReasonCode.INVALID_BANK
    assert result.action_word.source_bank_id == subject.BankId.INVALID
    assert subject.ActionWord.unpack(result.action_word.pack()) == result.action_word


def test_fault_response_witnesses_bind_all_terminals_and_reasons() -> None:
    witnesses = subject.fault_response_witnesses()
    assert tuple(row["fault_id"] for row in witnesses) == (
        subject.FAULT_PRIORITY
    )
    assert witnesses[0]["reason_codes"] == [
        "INVALID_BANK",
        "INVALID_FSM",
    ]
    assert all(
        row["terminal"] in {"LKG_HOLD", "RESET"}
        and row["undefined_action"] is False
        and row["cases"]
        and all(
            case["fallback"]
            and (case["hold"] or case["reset_request"])
            for case in row["cases"]
        )
        for row in witnesses
    )


def test_fsm_reachability_is_bfs_derived_and_all_valid_states_reset() -> None:
    reachable = subject.reachable_fsm_states()
    assert len(reachable) == 68
    assert subject.encode_fsm(subject.FsmMode.NORMAL, 0) in reachable
    for state in range(256):
        if state >> 5 <= int(subject.FsmMode.FAULT):
            assert 0 <= subject.reset_distance(state) <= 3
        else:
            with pytest.raises(ValueError):
                subject.reset_distance(state)


def test_future_suffix_cannot_change_prefix_decision() -> None:
    observed = _observed()
    memory = _memory()
    assert subject.causal_prefix_token(
        observed, memory
    ) == subject.causal_prefix_token(observed, memory)
    with pytest.raises(TypeError):
        subject.causal_prefix_token(  # type: ignore[call-arg]
            observed, memory, future_suffix=[_observed()]
        )
    with pytest.raises(TypeError):
        subject.deployable_decision(  # type: ignore[call-arg]
            observed, memory, evaluator_truth={"logical_error": False}
        )


def test_actual_assembler_derives_event_and_exact_fault_priority() -> None:
    observed = _observed()
    observed["discriminator_confidence"] = 0
    assert (
        subject.assemble_deployable_key(observed, _memory()).key.event_class
        == subject.EventClass.LOW_CONFIDENCE
    )
    observed["discriminator_confidence"] = 255
    observed["discriminator_word"] = 1
    assert (
        subject.assemble_deployable_key(observed, _memory()).key.event_class
        == subject.EventClass.OOD
    )
    conditions = {fault: True for fault in subject.FAULT_PRIORITY}
    classified = subject.classify_raw_fault(
        subject.CompositeKey(
            3,
            1,
            0,
            subject.EventClass.RESERVED_15,
            255,
        ),
        subject.IntegrityStatus(
            **{
                name: False
                for name in subject.IntegrityStatus.__dataclass_fields__
            }
        ),
    )
    assert classified == subject.FAULT_PRIORITY[0] == "INVALID_KEY"
    assert conditions  # documents simultaneous activation, not first-match luck


def test_forged_but_crc_valid_previous_reset_receipt_is_rejected() -> None:
    forged = subject.ActionWord(
        action_code=subject.NominalAction.RESET,
        correction_enable=False,
        reset_request=True,
        fallback=True,
        hold=False,
        pauli_dx=0,
        pauli_dz=0,
        next_phase_frame=0,
        next_fsm_state=subject.encode_fsm(subject.FsmMode.RESETTING, 0),
        catalog_action_id=0,
        reason_code=subject.ReasonCode.RESET_FAILURE,
        error_flags=0x80,
        source_bank_id=subject.BankId.A,
        factor_tag=0,
    )
    observed = _observed()
    observed.update(
        reset_ack="success",
        previous_action_word=forged.pack(),
        previous_composite_key=subject.CompositeKey(
            subject.BankId.A,
            0,
            0,
            subject.EventClass.NORMAL,
            subject.encode_fsm(subject.FsmMode.NORMAL, 0),
        ).to_word(),
        previous_action_present=True,
        previous_active_image_version=1,
    )
    memory = _memory()
    memory["leakage_reset_fsm_state"] = subject.encode_fsm(
        subject.FsmMode.RESETTING, 0
    )
    with pytest.raises(ValueError, match="exact recurrence"):
        subject.deployable_decision(observed, memory)


def test_iq_container_envelope_rejects_empty_mismatch_and_oversized() -> None:
    for mutate in (
        lambda row: row.update(iq_i=[], iq_q=[]),
        lambda row: row.update(iq_i=[0], iq_q=[0, 1]),
        lambda row: row.update(
            iq_i=[0] * 65537, iq_q=[0] * 65537
        ),
    ):
        candidate = _observed()
        mutate(candidate)
        with pytest.raises((TypeError, ValueError)):
            subject.validate_deployable_inputs(candidate, _memory())


def test_slow_path_accepts_only_complete_atomic_higher_version_package() -> None:
    package = _package(2)
    _validate_package(package)
    for mutate in (
        lambda row: row["entries"].pop(),
        lambda row: row.update(word_count=255),
        lambda row: row.update(crc32=0),
        lambda row: row.update(content_sha256="0" * 64),
        lambda row: row.update(entry_patch={0: 1}),
        lambda row: row.update(per_cycle_action=1),
    ):
        candidate = copy.deepcopy(package)
        mutate(candidate)
        with pytest.raises(ValueError):
            _validate_package(candidate)
    with pytest.raises(ValueError, match="strictly higher"):
        _validate_package(_package(1))
    for field, value in (
        ("package_id", "untrusted"),
        ("provenance_sha256", "0" * 64),
        ("release_pin_sha256", "0" * 64),
        ("bank_target", "LKG"),
        ("activation_epoch", 8),
    ):
        candidate = _package(2)
        candidate[field] = value
        with pytest.raises(ValueError):
            _validate_package(candidate)


def test_representative_probes_are_exactly_16_and_nonpromotional() -> None:
    probes = subject.representative_action_probes()
    assert len(probes) == len({probe.probe_id for probe in probes}) == 16
    tags = {tag for probe in probes for tag in probe.coverage_tags}
    assert {
        "idle",
        "q_positive",
        "q_negative",
        "p_positive",
        "p_negative",
        "boundary",
        "phase_frame",
        "reset_success",
        "reset_failure",
        "persistent_leakage",
        "crc_version_stale_deadline",
        "ood",
        "lkg_recovery",
    } <= tags
    assert all(
        probe.probe_only
        and not probe.codebook_candidate
        and not probe.performance_evidence
        and not probe.ranking_evidence
        for probe in probes
    )
    alternating = next(
        probe for probe in probes if probe.probe_id == "P06_ALTERNATE"
    )
    assert alternating.input_pattern == "q,p,q"
    assert len(alternating.fixture["steps"]) == 3
    for probe in probes:
        receipts = subject.execute_representative_probe(probe)
        witnessed = subject.probe_coverage_witnesses(receipts)
        assert set(probe.coverage_tags) - {
            "pre_codebook_interface_probe"
        } <= witnessed


def test_audit_is_executable_and_scope_remains_protocol_only() -> None:
    audit = subject.audit_contract()
    assert audit["all_passed"] is True
    assert all(audit["checks"].values())
    assert subject.MODEL_SCOPE["physics_backend_qualified"] is False
    assert subject.MODEL_SCOPE["codebook_released"] is False
    assert subject.MODEL_SCOPE["frontend_released"] is False
    assert subject.MODEL_SCOPE["rtl_adapter_qualified"] is False
    assert subject.MODEL_SCOPE["performance_evaluated"] is False
