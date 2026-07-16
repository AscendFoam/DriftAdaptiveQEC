"""T4.3.2 exhaustive transfer, atomic commit, hysteresis and readback validation."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.atomic_parameter_bank import (
    MODEL_SCOPE,
    AtomicParameterBankConfig,
    AtomicParameterBankError,
    AtomicParameterImageBank,
    ParameterImageManifest,
    build_parameter_image_manifest,
    seal_parameter_image_manifest,
    serialize_parameter_image,
    verify_commit_ack_readback,
)
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTInput,
    ParametricMAPLUTPipeline,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "t4_3_2_atomic_parameter_bank_validation.json"
DEFAULT_CSV = ROOT / "docs" / "t4_3_2_atomic_parameter_bank_source_data.csv"
SCHEMA_VERSION = "t4.3.2-atomic-parameter-bank-validation-v1"


def _images() -> tuple[Any, Any, Any]:
    config = ParametricMAPLUTConfig()
    profiles = registered_parameter_profiles(config)
    return tuple(
        compile_parametric_map_lut(
            profiles[index][0], active_bank_version=index, config=config
        )
        for index in range(3)
    )  # type: ignore[return-value]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/atomic_parameter_bank.py",
        "cnn_fpga/runtime/parametric_map_lut.py",
        "cnn_fpga/runtime/fast_path_fixed_point.py",
        "cnn_fpga/runtime/three_timescale_cadence.py",
        "cnn_fpga/benchmark/atomic_parameter_bank_validation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _row_hash(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    extra = sorted(set().union(*(set(row) for row in rows)) - set(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames + extra)
        writer.writeheader()
        writer.writerows(rows)


def _prime(
    bank: AtomicParameterImageBank,
    *,
    key: str = "regime-a",
    first_window: int = 1,
) -> None:
    bank.observe_selection(window_id=first_window, selection_key=key, eligible=True)
    state = bank.observe_selection(
        window_id=first_window + 1, selection_key=key, eligible=True
    )
    if not state.promotable:
        raise RuntimeError("failed to prime selection hysteresis")


def _manifest(
    bank: AtomicParameterImageBank,
    image: Any,
    *,
    transaction_id: str = "tx-v1",
    key: str = "regime-a",
    source_window_id: int = 2,
    created_epoch: int = 6048,
    apply_epoch: int = 6248,
) -> tuple[ParameterImageManifest, bytes]:
    return build_parameter_image_manifest(
        image,
        transaction_id=transaction_id,
        selection_key=key,
        expected_active_version=bank.active_version,
        source_window_id=source_window_id,
        created_epoch=created_epoch,
        apply_epoch=apply_epoch,
    )


def _stage_full(
    bank: AtomicParameterImageBank,
    manifest: ParameterImageManifest,
    payload: bytes,
    *,
    epoch: int,
    chunks: Sequence[tuple[int, bytes]] | None = None,
) -> None:
    bank.begin_stage(manifest, current_epoch=epoch)
    actual = [(0, payload)] if chunks is None else list(chunks)
    for offset, chunk in actual:
        bank.write_chunk(manifest.transaction_id, offset=offset, chunk=chunk)
    bank.finalize_stage(manifest.transaction_id, current_epoch=epoch)


def _base_row(
    family: str,
    case: str,
    *,
    index: int,
    observed: str,
    active_before: int = 0,
    active_after: int = 0,
    inactive_version: int | None = None,
    accepted: bool = False,
) -> dict[str, Any]:
    return {
        "scenario_family": family,
        "case": case,
        "index": index,
        "observed": observed,
        "active_version_before": active_before,
        "active_version_after": active_after,
        "inactive_slot_version": "" if inactive_version is None else inactive_version,
        "accepted": int(accepted),
    }


def _partial_cut_sweep(images: tuple[Any, Any, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = serialize_parameter_image(images[1])
    rows: list[dict[str, Any]] = []
    for cut in range(len(payload)):
        bank = AtomicParameterImageBank(images[0])
        _prime(bank)
        manifest, _ = _manifest(bank, images[1], transaction_id=f"partial-{cut}")
        bank.begin_stage(manifest, current_epoch=6048)
        if cut:
            bank.write_chunk(manifest.transaction_id, offset=0, chunk=payload[:cut])
        reason = "unexpected_accept"
        try:
            bank.finalize_stage(manifest.transaction_id, current_epoch=6048)
        except AtomicParameterBankError as exc:
            reason = exc.reason
        snapshot = bank.snapshot()
        rows.append(
            _base_row(
                "partial_cut",
                "prefix_missing_suffix",
                index=cut,
                observed=reason,
                active_after=snapshot["active_version"],
                inactive_version=snapshot["slot_versions"]["B"],
            )
        )
    return rows, {
        "payload_bytes": len(payload),
        "cutpoints_tested": len(rows),
        "all_rejected_incomplete": all(row["observed"] == "payload_incomplete" for row in rows),
        "all_active_unchanged": all(row["active_version_after"] == 0 for row in rows),
        "all_inactive_slots_unpublished": all(row["inactive_slot_version"] == "" for row in rows),
    }


def _single_byte_corruption_sweep(
    images: tuple[Any, Any, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = serialize_parameter_image(images[1])
    rows: list[dict[str, Any]] = []
    for index in range(len(payload)):
        bank = AtomicParameterImageBank(images[0])
        _prime(bank)
        manifest, _ = _manifest(bank, images[1], transaction_id=f"flip-{index}")
        corrupted = bytearray(payload)
        corrupted[index] ^= 1
        bank.begin_stage(manifest, current_epoch=6048)
        bank.write_chunk(manifest.transaction_id, offset=0, chunk=bytes(corrupted))
        reason = "unexpected_accept"
        try:
            bank.finalize_stage(manifest.transaction_id, current_epoch=6048)
        except AtomicParameterBankError as exc:
            reason = exc.reason
        snapshot = bank.snapshot()
        rows.append(
            _base_row(
                "single_byte_corruption",
                "xor_lsb",
                index=index,
                observed=reason,
                active_after=snapshot["active_version"],
                inactive_version=snapshot["slot_versions"]["B"],
            )
        )
    return rows, {
        "payload_bytes": len(payload),
        "byte_positions_tested": len(rows),
        "all_rejected_by_transfer_crc": all(
            row["observed"] == "transfer_crc_mismatch" for row in rows
        ),
        "all_active_unchanged": all(row["active_version_after"] == 0 for row in rows),
        "all_inactive_slots_unpublished": all(row["inactive_slot_version"] == "" for row in rows),
    }


def _chunk_and_commit_matrix(
    images: tuple[Any, Any, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = serialize_parameter_image(images[1])
    rows: list[dict[str, Any]] = []
    cases: list[tuple[str, list[tuple[int, bytes]]]] = []
    for size in (1, 7, 64, 511, len(payload)):
        chunks = [
            (offset, payload[offset : offset + size])
            for offset in range(0, len(payload), size)
        ]
        cases.append((f"forward_{size}", chunks))
        cases.append((f"reverse_{size}", list(reversed(chunks))))
    for case_index, (name, chunks) in enumerate(cases):
        bank = AtomicParameterImageBank(images[0])
        _prime(bank)
        manifest, _ = _manifest(bank, images[1], transaction_id=f"chunk-{case_index}")
        bank.begin_stage(manifest, current_epoch=6048)
        intermediate_immutable = True
        for offset, chunk in chunks:
            bank.write_chunk(manifest.transaction_id, offset=offset, chunk=chunk)
            state = bank.snapshot()
            intermediate_immutable &= (
                state["active_version"] == 0 and state["slot_versions"]["B"] is None
            )
        staged = bank.finalize_stage(manifest.transaction_id, current_epoch=6048)
        before = bank.commit_if_ready(6247, safe_boundary=True)
        deferred = bank.commit_if_ready(6248, safe_boundary=False)
        ack = bank.commit_if_ready(6249, safe_boundary=True)
        if ack is None:
            raise RuntimeError("happy-path commit produced no acknowledgement")
        readback = bank.readback(epoch=6249)
        confirmed = verify_commit_ack_readback(ack, readback)
        rows.append(
            _base_row(
                "chunk_commit",
                name,
                index=case_index,
                observed="commit_applied" if confirmed else "confirmation_failed",
                active_after=bank.active_version,
                inactive_version=0,
                accepted=ack.accepted,
            )
        )
        rows[-1].update(
            {
                "intermediate_immutable": int(intermediate_immutable),
                "staged_bank": staged.target_bank,
                "before_apply_none": int(before is None),
                "unsafe_deferred": int(
                    deferred is not None
                    and deferred.status == "deferred"
                    and not deferred.final
                ),
            }
        )
    return rows, {
        "cases": len(cases),
        "chunk_sizes": [1, 7, 64, 511, len(payload)],
        "all_intermediate_active_and_inactive_slots_immutable": all(
            row["intermediate_immutable"] for row in rows
        ),
        "all_committed_and_confirmed": all(
            row["observed"] == "commit_applied" and row["accepted"] == 1
            for row in rows
        ),
        "all_unsafe_boundaries_deferred": all(row["unsafe_deferred"] for row in rows),
    }


def _negative_scenarios(images: tuple[Any, Any, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(case: str, reason: str, bank: AtomicParameterImageBank) -> None:
        state = bank.snapshot()
        rows.append(
            _base_row(
                "negative_contract",
                case,
                index=len(rows),
                observed=reason,
                active_after=state["active_version"],
                inactive_version=state["slot_versions"][state["inactive_bank"]],
            )
        )

    # Hysteresis missing and key/source changes.
    bank = AtomicParameterImageBank(images[0])
    bank.observe_selection(window_id=1, selection_key="a", eligible=True)
    manifest, _ = _manifest(bank, images[1], key="a", source_window_id=1)
    try:
        bank.begin_stage(manifest, current_epoch=6048)
    except AtomicParameterBankError as exc:
        record("one_window_not_promotable", exc.reason, bank)

    bank = AtomicParameterImageBank(images[0])
    bank.observe_selection(window_id=1, selection_key="a", eligible=True)
    bank.observe_selection(window_id=2, selection_key="b", eligible=True)
    manifest, _ = _manifest(bank, images[1], key="b", source_window_id=2)
    try:
        bank.begin_stage(manifest, current_epoch=6048)
    except AtomicParameterBankError as exc:
        record("key_change_resets_run", exc.reason, bank)

    bank = AtomicParameterImageBank(images[0])
    _prime(bank)
    manifest, payload = _manifest(bank, images[1])
    _stage_full(bank, manifest, payload, epoch=6048)
    bank.observe_selection(window_id=3, selection_key="other", eligible=True)
    ack = bank.commit_if_ready(6248, safe_boundary=True)
    if ack is None:
        raise RuntimeError("invalidated hysteresis produced no ack")
    record("hysteresis_invalidated_after_finalize", ack.reason, bank)

    # Timestamp, CAS, version, digest, writer and freshness failures.
    for case, mutate, expected in (
        (
            "timestamp_epoch_mismatch",
            lambda item: replace(item, created_timestamp_ns=item.created_timestamp_ns + 1),
            "timestamp_epoch_mismatch",
        ),
        (
            "expected_active_version_mismatch",
            lambda item: replace(item, expected_active_version=1, new_version=2),
            "expected_active_version_mismatch",
        ),
        (
            "apply_epoch_stale",
            lambda item: replace(item, apply_epoch=item.created_epoch + 8193),
            "apply_epoch_stale",
        ),
    ):
        bank = AtomicParameterImageBank(images[0])
        _prime(bank)
        manifest, _ = _manifest(bank, images[1])
        try:
            bank.begin_stage(
                seal_parameter_image_manifest(mutate(manifest)), current_epoch=6048
            )
        except AtomicParameterBankError as exc:
            if exc.reason != expected:
                raise
            record(case, exc.reason, bank)

    for case, field, value, expected in (
        ("transfer_crc", "payload_crc32", "0" * 8, "transfer_crc_mismatch"),
        ("transfer_sha", "payload_sha256", "0" * 64, "transfer_sha256_mismatch"),
        ("manifest_image_sha", "image_sha256", "0" * 64, "manifest_image_digest_mismatch"),
    ):
        bank = AtomicParameterImageBank(images[0])
        _prime(bank)
        manifest, payload = _manifest(bank, images[1], transaction_id=case)
        changed = seal_parameter_image_manifest(replace(manifest, **{field: value}))
        bank.begin_stage(changed, current_epoch=6048)
        bank.write_chunk(changed.transaction_id, offset=0, chunk=payload)
        try:
            bank.finalize_stage(changed.transaction_id, current_epoch=6048)
        except AtomicParameterBankError as exc:
            if exc.reason != expected:
                raise
            record(case, exc.reason, bank)

    bank = AtomicParameterImageBank(images[0])
    _prime(bank)
    manifest, _ = _manifest(bank, images[1], transaction_id="manifest-bitflip")
    try:
        bank.begin_stage(replace(manifest, apply_epoch=6249), current_epoch=6048)
    except AtomicParameterBankError as exc:
        record("manifest_header_bitflip", exc.reason, bank)

    bank = AtomicParameterImageBank(images[0])
    _prime(bank)
    manifest, _ = _manifest(bank, images[1], transaction_id="manifest-sha-bitflip")
    try:
        bank.begin_stage(
            replace(manifest, manifest_sha256="0" * 64), current_epoch=6048
        )
    except AtomicParameterBankError as exc:
        record("manifest_sha256_bitflip", exc.reason, bank)

    bank = AtomicParameterImageBank(images[0])
    _prime(bank)
    manifest, payload = _manifest(bank, images[1], transaction_id="overlap")
    bank.begin_stage(manifest, current_epoch=6048)
    bank.write_chunk("overlap", offset=0, chunk=payload[:8])
    corrupt = bytes([payload[0] ^ 1]) + payload[1:8]
    try:
        bank.write_chunk("overlap", offset=0, chunk=corrupt)
    except AtomicParameterBankError as exc:
        record("conflicting_overlap", exc.reason, bank)
    try:
        bank.begin_stage(manifest, current_epoch=6048)
    except AtomicParameterBankError as exc:
        record("transaction_id_replay", exc.reason, bank)

    bank = AtomicParameterImageBank(images[0])
    _prime(bank)
    manifest, payload = _manifest(bank, images[1], transaction_id="stale-finalize")
    bank.begin_stage(manifest, current_epoch=6048)
    bank.write_chunk(manifest.transaction_id, offset=0, chunk=payload)
    try:
        bank.finalize_stage(manifest.transaction_id, current_epoch=14241)
    except AtomicParameterBankError as exc:
        record("stale_finalize", exc.reason, bank)

    bank = AtomicParameterImageBank(images[0])
    _prime(bank)
    manifest, payload = _manifest(bank, images[1], transaction_id="stale-commit")
    _stage_full(bank, manifest, payload, epoch=6048)
    ack = bank.commit_if_ready(14241, safe_boundary=True)
    if ack is None:
        raise RuntimeError("stale commit produced no ack")
    record("stale_commit", ack.reason, bank)

    return rows, {
        "cases": len(rows),
        "reasons": sorted({str(row["observed"]) for row in rows}),
        "all_active_unchanged": all(row["active_version_after"] == 0 for row in rows),
    }


def _double_bank_pipeline_and_race(
    images: tuple[Any, Any, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bank = AtomicParameterImageBank(images[0])
    _prime(bank)
    manifest, payload = _manifest(bank, images[1])
    _stage_full(bank, manifest, payload, epoch=6048)

    pipeline = ParametricMAPLUTPipeline(bank.read_active_image())
    pipeline.step(0, ParametricMAPLUTInput(0, 700, 0, 0))
    first_ack = bank.commit_if_ready(6248, safe_boundary=True)
    if first_ack is None or not first_ack.accepted:
        raise RuntimeError("first bank commit failed")
    pipeline.load_image(bank.read_active_image())
    pipeline.step(1, ParametricMAPLUTInput(1, 700, 0, 1))
    for cycle in range(2, 5):
        pipeline.step(cycle)
    old_output = pipeline.step(5)
    new_output = pipeline.step(6)
    first_readback = bank.readback(epoch=6248)
    first_confirmed = verify_commit_ack_readback(first_ack, first_readback)
    rows.append(
        _base_row(
            "double_bank_pipeline",
            "A_to_B",
            index=0,
            observed="commit_applied",
            active_after=bank.active_version,
            inactive_version=0,
            accepted=first_ack.accepted,
        )
    )

    _prime(bank, key="regime-c", first_window=3)
    second_manifest, second_payload = _manifest(
        bank,
        images[2],
        transaction_id="tx-v2",
        key="regime-c",
        source_window_id=4,
        created_epoch=14048,
        apply_epoch=14248,
    )
    _stage_full(bank, second_manifest, second_payload, epoch=14048)
    before_second = bank.snapshot()
    second_ack = bank.commit_if_ready(14248, safe_boundary=True)
    if second_ack is None or not second_ack.accepted:
        raise RuntimeError("second bank commit failed")
    second_confirmed = verify_commit_ack_readback(
        second_ack, bank.readback(epoch=14248)
    )
    rows.append(
        _base_row(
            "double_bank_pipeline",
            "B_to_A",
            index=1,
            observed="commit_applied",
            active_before=1,
            active_after=bank.active_version,
            inactive_version=1,
            accepted=second_ack.accepted,
        )
    )

    race_bank = AtomicParameterImageBank(images[0])
    _prime(race_bank)
    race_manifest, _ = _manifest(race_bank, images[1], transaction_id="race")

    def attempt() -> str:
        try:
            race_bank.begin_stage(race_manifest, current_epoch=6048)
            return "accepted"
        except AtomicParameterBankError as exc:
            return exc.reason

    with ThreadPoolExecutor(max_workers=2) as pool:
        race_results = list(pool.map(lambda _: attempt(), range(2)))
    rows.append(
        _base_row(
            "double_bank_pipeline",
            "concurrent_writer_race",
            index=2,
            observed="|".join(sorted(race_results)),
            active_after=race_bank.active_version,
        )
    )

    summary = {
        "first_commit_confirmed": first_confirmed,
        "second_commit_confirmed": second_confirmed,
        "bank_sequence": ["A:v0", "B:v1", "A:v2"],
        "second_stage_preserved_active_B_v1": before_second["active_bank"] == "B"
        and before_second["active_version"] == 1
        and before_second["slot_versions"] == {"A": 2, "B": 1},
        "inflight_versions": [
            None if old_output is None else old_output.active_bank_version,
            None if new_output is None else new_output.active_bank_version,
        ],
        "race_results": sorted(race_results),
    }
    return rows, summary


def _collect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = _images()
    partial_rows, partial = _partial_cut_sweep(images)
    corruption_rows, corruption = _single_byte_corruption_sweep(images)
    chunk_rows, chunk = _chunk_and_commit_matrix(images)
    negative_rows, negative = _negative_scenarios(images)
    integration_rows, integration = _double_bank_pipeline_and_race(images)
    rows = partial_rows + corruption_rows + chunk_rows + negative_rows + integration_rows
    return rows, {
        "image_versions": [image.active_bank_version for image in images],
        "image_sha256": [image.image_sha256 for image in images],
        "payload_lengths": [len(serialize_parameter_image(image)) for image in images],
        "partial_cut_sweep": partial,
        "single_byte_corruption_sweep": corruption,
        "chunk_commit_matrix": chunk,
        "negative_scenarios": negative,
        "double_bank_pipeline_and_race": integration,
    }


def _gate(gate_id: str, description: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"id": gate_id, "description": description, "passed": bool(passed), "evidence": evidence}


def run_validation(
    *,
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
) -> dict[str, Any]:
    rows_a, summary_a = _collect()
    rows_b, summary_b = _collect()
    partial = summary_a["partial_cut_sweep"]
    corrupt = summary_a["single_byte_corruption_sweep"]
    chunk = summary_a["chunk_commit_matrix"]
    negative = summary_a["negative_scenarios"]
    integration = summary_a["double_bank_pipeline_and_race"]
    manifest_fields = set(ParameterImageManifest.__dataclass_fields__)
    online_ast = ast.parse(
        textwrap.dedent(inspect.getsource(AtomicParameterImageBank.commit_if_ready))
    )
    gates = [
        _gate(
            "G01",
            "manifest contains schema, version/CAS, timestamp, source, apply and dual integrity fields",
            {
                "schema_version",
                "transaction_id",
                "selection_key",
                "expected_active_version",
                "new_version",
                "source_window_id",
                "created_epoch",
                "created_timestamp_ns",
                "apply_epoch",
                "payload_length",
                "payload_crc32",
                "payload_sha256",
                "image_crc32",
                "image_sha256",
                "manifest_crc32",
                "manifest_sha256",
            }
            <= manifest_fields,
            sorted(manifest_fields),
        ),
        _gate(
            "G02",
            "every proper prefix cut leaves both valid slots unpublished and active v0",
            partial["all_rejected_incomplete"]
            and partial["all_active_unchanged"]
            and partial["all_inactive_slots_unpublished"]
            and partial["cutpoints_tested"] == partial["payload_bytes"],
            partial,
        ),
        _gate(
            "G03",
            "flipping each payload byte once is rejected before image publication",
            corrupt["all_rejected_by_transfer_crc"]
            and corrupt["all_active_unchanged"]
            and corrupt["all_inactive_slots_unpublished"]
            and corrupt["byte_positions_tested"] == corrupt["payload_bytes"],
            corrupt,
        ),
        _gate(
            "G04",
            "forward/reverse chunk sizes including byte-at-a-time all produce one exact image",
            chunk["cases"] == 10 and chunk["all_committed_and_confirmed"],
            chunk,
        ),
        _gate(
            "G05",
            "active and valid inactive slots remain immutable throughout partial writes",
            chunk["all_intermediate_active_and_inactive_slots_immutable"],
            chunk,
        ),
        _gate(
            "G06",
            "apply epoch and unsafe boundary cannot activate early",
            chunk["all_unsafe_boundaries_deferred"]
            and all(row.get("before_apply_none") == 1 for row in rows_a if row["scenario_family"] == "chunk_commit"),
            chunk,
        ),
        _gate(
            "G07",
            "selection hysteresis requires two consecutive eligible same-key windows",
            "hysteresis_not_satisfied" in negative["reasons"]
            and "hysteresis_invalidated" in negative["reasons"],
            negative,
        ),
        _gate(
            "G08",
            "timestamp, CAS, apply age, digest, overlap and stale failures are reason coded",
            {
                "timestamp_epoch_mismatch",
                "expected_active_version_mismatch",
                "apply_epoch_stale",
                "transfer_crc_mismatch",
                "transfer_sha256_mismatch",
                "manifest_image_digest_mismatch",
                "manifest_crc_mismatch",
                "manifest_sha256_mismatch",
                "conflicting_overlap",
                "transaction_replay",
                "payload_stale",
                "payload_stale_before_commit",
            }
            <= set(negative["reasons"]),
            negative,
        ),
        _gate(
            "G09",
            "all negative contract scenarios preserve the active version",
            negative["all_active_unchanged"],
            negative,
        ),
        _gate(
            "G10",
            "ack is not success until CRC/SHA/version/bank readback matches",
            integration["first_commit_confirmed"] and integration["second_commit_confirmed"],
            integration,
        ),
        _gate(
            "G11",
            "two complete updates alternate A-v0 to B-v1 to A-v2 without active overwrite",
            integration["bank_sequence"] == ["A:v0", "B:v1", "A:v2"]
            and integration["second_stage_preserved_active_B_v1"],
            integration,
        ),
        _gate(
            "G12",
            "T4.2 pipeline latches old in-flight image while new requests use committed image",
            integration["inflight_versions"] == [0, 1],
            integration,
        ),
        _gate(
            "G13",
            "concurrent writers are serialized with exactly one accepted transfer",
            integration["race_results"]
            == ["accepted", "writer_conflict_transfer_in_progress"],
            integration["race_results"],
        ),
        _gate(
            "G14",
            "production config keeps two-window promotion, 4000-cycle residency and 8192-cycle age",
            asdict(AtomicParameterBankConfig())["promotion_good_windows"] == 2
            and asdict(AtomicParameterBankConfig())["min_residency_cycles"] == 4000
            and asdict(AtomicParameterBankConfig())["max_payload_age_cycles"] == 8192,
            asdict(AtomicParameterBankConfig()),
        ),
        _gate(
            "G15",
            "commit kernel contains no hidden-truth field access or payload mutation loop",
            not any(
                isinstance(node, ast.Name) and node.id in {"truth", "target_params", "hidden_state"}
                for node in ast.walk(online_ast)
            )
            and not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(online_ast)),
            {"for_or_while_nodes": sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(online_ast))},
        ),
        _gate(
            "G16",
            "independent full evidence reruns are row and summary deterministic",
            _row_hash(rows_a) == _row_hash(rows_b) and summary_a == summary_b,
            {"run_a": _row_hash(rows_a), "run_b": _row_hash(rows_b)},
        ),
        _gate(
            "G17",
            "evidence scope remains software contract rather than RTL or board measurement",
            AtomicParameterBankConfig().model_scope == MODEL_SCOPE
            and "not_rtl_or_board" in MODEL_SCOPE,
            MODEL_SCOPE,
        ),
    ]
    if not all(gate["passed"] for gate in gates):
        failed = [gate["id"] for gate in gates if not gate["passed"]]
        raise RuntimeError(f"T4.3.2 validation gates failed: {failed}")

    _write_csv(rows_a, csv_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "T4.3.2",
        "snapshot_date": "2026-07-15",
        "status": "PASS",
        "scope": MODEL_SCOPE,
        "config": asdict(AtomicParameterBankConfig()),
        "manifest_fields": sorted(manifest_fields),
        "summary": summary_a,
        "source_data": {
            "path": str(csv_path.resolve().relative_to(ROOT)).replace("\\", "/")
            if csv_path.resolve().is_relative_to(ROOT)
            else str(csv_path.resolve()).replace("\\", "/"),
            "rows": len(rows_a),
            "sha256": _sha256_file(csv_path),
            "row_sha256": _row_hash(rows_a),
        },
        "implementation_sha256": _implementation_sha256(),
        "gates": gates,
        "claim_boundary": {
            "allowed": "transactional double-bank software contract with exhaustive transfer-fault evidence",
            "forbidden": [
                "RTL atomicity",
                "FPGA or board commit timing",
                "device-calibrated hysteresis",
                "automatic physical rollback",
            ],
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    result = run_validation(json_path=args.json, csv_path=args.csv)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates_passed": sum(gate["passed"] for gate in result["gates"]),
                "gates_total": len(result["gates"]),
                "source_rows": result["source_data"]["rows"],
                "summary": result["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
