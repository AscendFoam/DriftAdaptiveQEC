from __future__ import annotations

import hashlib
import json
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.atomic_parameter_bank import (
    MODEL_SCOPE,
    ActiveImageReadback,
    AtomicParameterBankConfig,
    AtomicParameterBankError,
    AtomicParameterImageBank,
    build_parameter_image_manifest,
    deserialize_parameter_image,
    seal_parameter_image_manifest,
    serialize_parameter_image,
    verify_commit_ack_readback,
)
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTInput,
    ParametricMAPLUTPipeline,
)


def _images():
    config = ParametricMAPLUTConfig()
    profiles = registered_parameter_profiles(config)
    return tuple(
        compile_parametric_map_lut(profiles[index][0], active_bank_version=index, config=config)
        for index in range(3)
    )


def _small_config(**overrides) -> AtomicParameterBankConfig:
    values = {
        "promotion_good_windows": 2,
        "min_residency_cycles": 1,
        "max_payload_age_cycles": 100,
    }
    values.update(overrides)
    return AtomicParameterBankConfig(**values)


def _ready_manifest(
    bank: AtomicParameterImageBank,
    image,
    *,
    transaction_id: str = "tx-1",
    selection_key: str = "regime-a",
    source_window_id: int = 2,
    created_epoch: int = 10,
    apply_epoch: int = 12,
):
    bank.observe_selection(window_id=source_window_id - 1, selection_key=selection_key, eligible=True)
    state = bank.observe_selection(
        window_id=source_window_id, selection_key=selection_key, eligible=True
    )
    assert state.promotable
    return build_parameter_image_manifest(
        image,
        transaction_id=transaction_id,
        selection_key=selection_key,
        expected_active_version=bank.active_version,
        source_window_id=source_window_id,
        created_epoch=created_epoch,
        apply_epoch=apply_epoch,
    )


def _stage_complete(
    bank: AtomicParameterImageBank,
    manifest,
    payload: bytes,
    *,
    current_epoch: int,
    chunk_size: int = 137,
):
    bank.begin_stage(manifest, current_epoch=current_epoch)
    for offset in range(0, len(payload), chunk_size):
        bank.write_chunk(
            manifest.transaction_id,
            offset=offset,
            chunk=payload[offset : offset + chunk_size],
        )
    return bank.finalize_stage(manifest.transaction_id, current_epoch=current_epoch)


def test_image_serialization_is_canonical_and_self_verified() -> None:
    image = _images()[0]
    payload = serialize_parameter_image(image)
    restored = deserialize_parameter_image(payload)

    assert restored == image
    assert serialize_parameter_image(restored) == payload
    assert json.loads(payload)["active_bank_version"] == 0


def test_noncanonical_or_schema_extended_payload_is_rejected() -> None:
    image = _images()[0]
    decoded = json.loads(serialize_parameter_image(image))
    decoded["unexpected"] = 1
    payload = json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(AtomicParameterBankError, match="payload_schema_mismatch"):
        deserialize_parameter_image(payload)


def test_manifest_binds_transfer_image_version_timestamp_and_source_window() -> None:
    bank = AtomicParameterImageBank(_images()[0], _small_config())
    manifest, payload = _ready_manifest(bank, _images()[1])

    assert manifest.expected_active_version == 0
    assert manifest.new_version == 1
    assert manifest.created_timestamp_ns == 50_000
    assert manifest.source_window_id == 2
    assert manifest.payload_length == len(payload)
    assert manifest.payload_crc32 == f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    assert manifest.payload_sha256 == hashlib.sha256(payload).hexdigest()
    manifest.verify()
    assert len(manifest.manifest_crc32) == 8
    assert len(manifest.manifest_sha256) == 64


def test_selection_hysteresis_requires_consecutive_same_key_windows() -> None:
    bank = AtomicParameterImageBank(_images()[0], _small_config())
    first = bank.observe_selection(window_id=1, selection_key="a", eligible=True)
    switched = bank.observe_selection(window_id=2, selection_key="b", eligible=True)
    promoted = bank.observe_selection(window_id=3, selection_key="b", eligible=True)

    assert first.good_run == 1 and not first.promotable
    assert switched.good_run == 1 and not switched.promotable
    assert promoted.promotable
    assert promoted.evidence_window_ids == (2, 3)


def test_ineligible_window_resets_hysteresis_and_sequence_is_strict() -> None:
    bank = AtomicParameterImageBank(_images()[0], _small_config())
    bank.observe_selection(window_id=1, selection_key="a", eligible=True)
    reset = bank.observe_selection(window_id=2, selection_key="a", eligible=False)
    assert reset.good_run == 0 and not reset.promotable
    with pytest.raises(AtomicParameterBankError, match="window_sequence_nonmonotonic"):
        bank.observe_selection(window_id=2, selection_key="a", eligible=True)


def test_begin_rejects_missing_hysteresis_cas_and_bad_timestamp() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, _ = build_parameter_image_manifest(
        images[1],
        transaction_id="tx",
        selection_key="a",
        expected_active_version=0,
        source_window_id=1,
        created_epoch=10,
        apply_epoch=12,
    )
    with pytest.raises(AtomicParameterBankError, match="hysteresis_not_satisfied"):
        bank.begin_stage(manifest, current_epoch=10)

    manifest, _ = _ready_manifest(bank, images[1], selection_key="a")
    with pytest.raises(AtomicParameterBankError, match="timestamp_epoch_mismatch"):
        bank.begin_stage(
            seal_parameter_image_manifest(replace(manifest, created_timestamp_ns=1)),
            current_epoch=10,
        )
    with pytest.raises(AtomicParameterBankError, match="expected_active_version_mismatch"):
        bank.begin_stage(
            seal_parameter_image_manifest(
                replace(manifest, expected_active_version=1, new_version=2)
            ),
            current_epoch=10,
        )


def test_manifest_integrity_and_transaction_replay_fail_before_transfer() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    with pytest.raises(AtomicParameterBankError, match="manifest_crc_mismatch"):
        bank.begin_stage(replace(manifest, apply_epoch=13), current_epoch=10)
    with pytest.raises(AtomicParameterBankError, match="manifest_sha256_mismatch"):
        bank.begin_stage(
            replace(manifest, manifest_sha256="0" * 64), current_epoch=10
        )

    bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk(manifest.transaction_id, offset=0, chunk=payload[:8])
    corrupt = bytes([payload[0] ^ 1]) + payload[1:8]
    with pytest.raises(AtomicParameterBankError, match="conflicting_overlap"):
        bank.write_chunk(manifest.transaction_id, offset=0, chunk=corrupt)
    with pytest.raises(AtomicParameterBankError, match="transaction_replay"):
        bank.begin_stage(manifest, current_epoch=10)


def test_partial_transfer_never_changes_valid_bank_or_active_image() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk("tx-1", offset=0, chunk=payload[: len(payload) // 2])

    with pytest.raises(AtomicParameterBankError, match="payload_incomplete"):
        bank.finalize_stage("tx-1", current_epoch=10)
    snapshot = bank.snapshot()
    assert snapshot["active_version"] == 0
    assert snapshot["slot_versions"] == {"A": 0, "B": None}
    assert snapshot["pending_version"] is None


def test_out_of_order_chunks_and_idempotent_retransmit_finalize_exact_image() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    middle = len(payload) // 2
    bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk("tx-1", offset=middle, chunk=payload[middle:])
    bank.write_chunk("tx-1", offset=middle, chunk=payload[middle:])
    bank.write_chunk("tx-1", offset=0, chunk=payload[:middle])
    staged = bank.finalize_stage("tx-1", current_epoch=10)

    assert staged.target_bank == "B"
    assert staged.image_sha256 == images[1].image_sha256
    assert bank.active_version == 0
    assert bank.snapshot()["slot_versions"] == {"A": 0, "B": 1}


def test_conflicting_overlap_and_out_of_bounds_abort_transaction() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk("tx-1", offset=0, chunk=payload[:16])
    corrupt = bytes([payload[0] ^ 1]) + payload[1:16]
    with pytest.raises(AtomicParameterBankError, match="conflicting_overlap"):
        bank.write_chunk("tx-1", offset=0, chunk=corrupt)
    assert bank.snapshot()["transfer"] is None

    second = seal_parameter_image_manifest(replace(manifest, transaction_id="tx-2"))
    bank.begin_stage(second, current_epoch=10)
    with pytest.raises(AtomicParameterBankError, match="chunk_out_of_bounds"):
        bank.write_chunk("tx-2", offset=len(payload) - 1, chunk=b"xx")
    assert bank.snapshot()["transfer"] is None


def test_transfer_crc_sha_and_manifest_image_digests_fail_independently() -> None:
    images = _images()
    for field, value, reason in (
        ("payload_crc32", "00000000", "transfer_crc_mismatch"),
        ("payload_sha256", "0" * 64, "transfer_sha256_mismatch"),
        ("image_sha256", "0" * 64, "manifest_image_digest_mismatch"),
    ):
        bank = AtomicParameterImageBank(images[0], _small_config())
        manifest, payload = _ready_manifest(bank, images[1])
        changed = seal_parameter_image_manifest(replace(manifest, **{field: value}))
        bank.begin_stage(changed, current_epoch=10)
        bank.write_chunk(changed.transaction_id, offset=0, chunk=payload)
        with pytest.raises(AtomicParameterBankError, match=reason):
            bank.finalize_stage(changed.transaction_id, current_epoch=10)
        assert bank.active_version == 0


def test_single_byte_payload_corruption_never_reaches_image_decoder() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    corrupted = bytearray(payload)
    corrupted[len(corrupted) // 2] ^= 1
    bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk("tx-1", offset=0, chunk=bytes(corrupted))
    with pytest.raises(AtomicParameterBankError, match="transfer_crc_mismatch"):
        bank.finalize_stage("tx-1", current_epoch=10)
    assert bank.read_active_image() == images[0]


def test_transfer_and_pending_writer_conflicts_are_explicit() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    bank.begin_stage(manifest, current_epoch=10)
    with pytest.raises(AtomicParameterBankError, match="transfer_in_progress"):
        bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk("tx-1", offset=0, chunk=payload)
    bank.finalize_stage("tx-1", current_epoch=10)
    with pytest.raises(AtomicParameterBankError, match="pending_commit"):
        bank.begin_stage(manifest, current_epoch=10)


def test_concurrent_begin_has_exactly_one_winner() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, _ = _ready_manifest(bank, images[1])

    def attempt() -> str:
        try:
            bank.begin_stage(manifest, current_epoch=10)
            return "accepted"
        except AtomicParameterBankError as exc:
            return exc.reason

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert results.count("accepted") == 1
    assert results.count("writer_conflict_transfer_in_progress") == 1


def test_commit_waits_for_apply_epoch_and_safe_boundary_then_ack_matches_readback() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    _stage_complete(bank, manifest, payload, current_epoch=10)

    assert bank.commit_if_ready(11, safe_boundary=True) is None
    deferred = bank.commit_if_ready(12, safe_boundary=False)
    assert deferred is not None
    assert deferred.status == "deferred" and not deferred.final
    assert bank.active_version == 0
    ack = bank.commit_if_ready(13, safe_boundary=True)
    assert ack is not None and ack.accepted and ack.status == "committed"
    readback = bank.readback(epoch=13)
    assert verify_commit_ack_readback(ack, readback)
    assert readback.active_version == 1 and readback.active_bank == "B"


def test_minimum_residency_defers_without_dropping_verified_candidate() -> None:
    images = _images()
    config = _small_config(min_residency_cycles=20, max_payload_age_cycles=100)
    bank = AtomicParameterImageBank(images[0], config)
    manifest, payload = _ready_manifest(bank, images[1])
    _stage_complete(bank, manifest, payload, current_epoch=10)

    deferred = bank.commit_if_ready(12, safe_boundary=True)
    assert deferred is not None and deferred.reason == "minimum_residency_not_met"
    assert bank.pending is not None and bank.active_version == 0
    ack = bank.commit_if_ready(20, safe_boundary=True)
    assert ack is not None and ack.accepted


def test_selection_change_after_finalize_invalidates_pending_commit() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    _stage_complete(bank, manifest, payload, current_epoch=10)
    bank.observe_selection(window_id=3, selection_key="regime-b", eligible=True)
    ack = bank.commit_if_ready(12, safe_boundary=True)

    assert ack is not None and ack.status == "rejected"
    assert ack.reason == "hysteresis_invalidated"
    assert bank.active_version == 0 and bank.pending is None
    assert bank.snapshot()["slot_versions"]["B"] is None


def test_stale_finalize_and_stale_commit_fail_closed() -> None:
    images = _images()
    config = _small_config(max_payload_age_cycles=20)
    bank = AtomicParameterImageBank(images[0], config)
    manifest, payload = _ready_manifest(
        bank, images[1], created_epoch=10, apply_epoch=20
    )
    bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk("tx-1", offset=0, chunk=payload)
    with pytest.raises(AtomicParameterBankError, match="payload_stale"):
        bank.finalize_stage("tx-1", current_epoch=31)
    assert bank.active_version == 0

    bank = AtomicParameterImageBank(images[0], config)
    manifest, payload = _ready_manifest(
        bank, images[1], created_epoch=10, apply_epoch=20
    )
    _stage_complete(bank, manifest, payload, current_epoch=10)
    ack = bank.commit_if_ready(31, safe_boundary=True)
    assert ack is not None and ack.reason == "payload_stale_before_commit"
    assert bank.active_version == 0


def test_double_bank_alternates_only_after_each_complete_commit() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    first_manifest, first_payload = _ready_manifest(bank, images[1])
    _stage_complete(bank, first_manifest, first_payload, current_epoch=10)
    first_ack = bank.commit_if_ready(12, safe_boundary=True)
    assert first_ack is not None and first_ack.active_bank == "B"

    second_manifest, second_payload = _ready_manifest(
        bank,
        images[2],
        transaction_id="tx-2",
        selection_key="regime-c",
        source_window_id=4,
        created_epoch=13,
        apply_epoch=15,
    )
    _stage_complete(bank, second_manifest, second_payload, current_epoch=13)
    assert bank.snapshot()["slot_versions"] == {"A": 2, "B": 1}
    assert bank.active_bank == "B" and bank.active_version == 1
    second_ack = bank.commit_if_ready(15, safe_boundary=True)
    assert second_ack is not None and second_ack.active_bank == "A"
    assert bank.active_version == 2


def test_pipeline_latches_old_image_for_inflight_request_across_atomic_commit() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    pipeline = ParametricMAPLUTPipeline(bank.read_active_image())
    assert pipeline.step(0, ParametricMAPLUTInput(0, 700, 0, 0)) is None

    manifest, payload = _ready_manifest(bank, images[1])
    _stage_complete(bank, manifest, payload, current_epoch=10)
    ack = bank.commit_if_ready(12, safe_boundary=True)
    assert ack is not None and ack.accepted
    pipeline.load_image(bank.read_active_image())
    assert pipeline.step(1, ParametricMAPLUTInput(1, 700, 0, 1)) is None
    for cycle in range(2, 5):
        assert pipeline.step(cycle) is None
    old = pipeline.step(5)
    new = pipeline.step(6)
    assert old is not None and old.active_bank_version == 0
    assert new is not None and new.active_bank_version == 1


def test_ack_readback_mismatch_and_noncommitted_ack_are_rejected() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    _stage_complete(bank, manifest, payload, current_epoch=10)
    deferred = bank.commit_if_ready(12, safe_boundary=False)
    assert deferred is not None
    with pytest.raises(AtomicParameterBankError, match="ack_not_committed"):
        verify_commit_ack_readback(deferred, bank.readback())
    ack = bank.commit_if_ready(13, safe_boundary=True)
    assert ack is not None
    readback = bank.readback()
    with pytest.raises(AtomicParameterBankError, match="readback_mismatch"):
        verify_commit_ack_readback(
            ack, replace(readback, image_sha256="0" * 64)
        )


def test_snapshot_exposes_coverage_not_partial_payload_bytes() -> None:
    images = _images()
    bank = AtomicParameterImageBank(images[0], _small_config())
    manifest, payload = _ready_manifest(bank, images[1])
    bank.begin_stage(manifest, current_epoch=10)
    bank.write_chunk("tx-1", offset=0, chunk=payload[:11])
    snapshot = bank.snapshot()

    assert snapshot["model_scope"] == MODEL_SCOPE
    assert snapshot["transfer"]["written_bytes"] == 11
    assert "buffer" not in snapshot["transfer"]
    assert snapshot["active_version"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"promotion_good_windows": 0},
        {"min_residency_cycles": 0},
        {"max_payload_age_cycles": 3999},
        {"max_payload_age_cycles": 65536},
        {"safe_boundary_period_cycles": 0},
    ],
)
def test_invalid_bank_configs_fail_closed(overrides: dict[str, int]) -> None:
    with pytest.raises((TypeError, ValueError)):
        AtomicParameterBankConfig(**overrides)


def test_readback_type_is_immutable_record() -> None:
    bank = AtomicParameterImageBank(_images()[0], _small_config())
    readback = bank.readback()
    assert isinstance(readback, ActiveImageReadback)
    with pytest.raises(Exception):
        readback.active_version = 7  # type: ignore[misc]


def test_atomic_bank_is_available_through_runtime_public_api() -> None:
    from cnn_fpga.runtime import (
        AtomicParameterBankConfig as ExportedConfig,
        AtomicParameterImageBank as ExportedBank,
        ThreeTimescaleCadence as ExportedCadence,
    )

    assert ExportedConfig is AtomicParameterBankConfig
    assert ExportedBank is AtomicParameterImageBank
    assert ExportedCadence.__module__ == "cnn_fpga.runtime.three_timescale_cadence"
