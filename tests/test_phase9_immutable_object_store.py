from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from cnn_fpga.benchmark.phase9_immutable_object_store import (
    ImmutableObjectStore,
    _sha,
    append_attempt_event,
    publish_inventory_and_manifest,
)

SOURCE_SHA = "3" * 64
SEED_NAMESPACE = "fixture"
RUNNER_ID = "fixture_runner"


def _fingerprint(**extra: object) -> dict[str, object]:
    fingerprint: dict[str, object] = {
        "runner_id": RUNNER_ID,
        "python": [3, 12, 7],
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "psutil": "5.9.0",
        "platform": "test-platform",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "seed_namespace": SEED_NAMESPACE,
    }
    fingerprint.update(extra)
    return fingerprint


def _store(tmp_path: Path) -> ImmutableObjectStore:
    return ImmutableObjectStore(
        repository_root=tmp_path,
        object_root=tmp_path / "run/objects/sha256",
        staging_root=tmp_path / "run/staging",
        receipt_root=tmp_path / "run/receipts",
        task_id="T-RISK-20260728-04",
        run_id="fresh-test-run",
        config_sha256="1" * 64,
        plan_sha256="2" * 64,
        source_snapshot_sha256=SOURCE_SHA,
        seed_namespace=SEED_NAMESPACE,
        runner_id=RUNNER_ID,
    )


def _cell(chunk_id: str, index: int = 0) -> dict[str, object]:
    return {
        "plan_index": index,
        "chunk_id": chunk_id,
        "layer": "shared",
        "cutoff": 44,
        "backend": "B",
        "expected_rows": 2,
    }


def _receipt(
    store: ImmutableObjectStore,
    cell: dict[str, object],
    payload: bytes,
) -> dict[str, object]:
    binding = store.put_bytes(
        payload,
        role="round_ledger_csv",
        media_type="text/csv",
    )
    return store.commit_receipt(
        cell=cell,
        objects=[binding],
        expected_rows=2,
        observed_rows=2,
        exception_rows=0,
        missing_rows=0,
        conservation_failures=0,
        source_snapshot_sha256=SOURCE_SHA,
        runtime_fingerprint=_fingerprint(),
        reset_rows=0,
        reset_sidecar_rows=0,
    )


def test_object_is_content_addressed_reopened_and_deduplicated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.put_bytes(b"same bytes", role="a")
    second = store.put_bytes(b"same bytes", role="b")
    assert first.sha256 == second.sha256
    assert first.path == second.path
    assert first.reopened_and_rehashed is True
    assert first.file_fsync is True
    objects = list((tmp_path / "run/objects/sha256").glob("*/*"))
    assert len(objects) == 1
    assert not list((tmp_path / "run/staging").iterdir())


def test_orphan_object_has_no_receipt_and_no_inventory_vote(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.put_bytes(b"orphan", role="orphan")
    assert not list((tmp_path / "run/receipts").glob("*.json"))
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        store.inventory([_cell("cell0")])


def test_receipt_is_idempotent_but_conflicting_replay_fails(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    first = _receipt(store, cell, b"rows")
    binding = store.put_bytes(b"rows", role="round_ledger_csv", media_type="text/csv")
    second = store.commit_receipt(
        cell=cell,
        objects=[binding],
        expected_rows=2,
        observed_rows=2,
        exception_rows=0,
        missing_rows=0,
        conservation_failures=0,
        source_snapshot_sha256=SOURCE_SHA,
        runtime_fingerprint=_fingerprint(),
        reset_rows=0,
        reset_sidecar_rows=0,
    )
    assert first == second
    other = store.put_bytes(b"different", role="round_ledger_csv", media_type="text/csv")
    with pytest.raises(RuntimeError, match="conflicting immutable receipt"):
        store.commit_receipt(
            cell=cell,
            objects=[other],
            expected_rows=2,
            observed_rows=2,
            exception_rows=0,
            missing_rows=0,
            conservation_failures=0,
            source_snapshot_sha256=SOURCE_SHA,
            runtime_fingerprint=_fingerprint(),
            reset_rows=0,
            reset_sidecar_rows=0,
        )


def test_concurrent_object_and_conflicting_receipt_publication_is_exclusive(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        bindings = list(
            pool.map(
                lambda role: store.put_bytes(b"same", role=role),
                ("r0", "r1", "r2", "r3"),
            )
        )
    assert len({item.sha256 for item in bindings}) == 1
    assert len(list((tmp_path / "run/objects/sha256").glob("*/*"))) == 1

    first = store.put_bytes(b"first", role="round_ledger_csv")
    second = store.put_bytes(b"second", role="round_ledger_csv")
    barrier = Barrier(2)

    def publish(binding: object) -> object:
        barrier.wait()
        return store.commit_receipt(
            cell=_cell("raced"),
            objects=[binding],
            expected_rows=2,
            observed_rows=2,
            exception_rows=0,
            missing_rows=0,
            conservation_failures=0,
            source_snapshot_sha256=SOURCE_SHA,
            runtime_fingerprint=_fingerprint(),
            reset_rows=0,
            reset_sidecar_rows=0,
        )

    outcomes: list[object] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, item) for item in (first, second)]
        for future in futures:
            try:
                outcomes.append(future.result())
            except RuntimeError as exc:
                outcomes.append(exc)
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert sum(isinstance(item, RuntimeError) for item in outcomes) == 1
    store.verify_receipt(
        store.receipt_path("raced"),
        expected_cell=_cell("raced"),
    )


def test_receipt_denominator_and_reset_invariants_are_enforced(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    binding = store.put_bytes(b"rows", role="round_ledger_csv")
    with pytest.raises(ValueError, match="denominator"):
        store.commit_receipt(
            cell=_cell("bad-denominator"),
            objects=[binding],
            expected_rows=2,
            observed_rows=1,
            exception_rows=0,
            missing_rows=0,
            conservation_failures=0,
            source_snapshot_sha256=SOURCE_SHA,
            runtime_fingerprint={},
            reset_rows=0,
            reset_sidecar_rows=0,
        )
    with pytest.raises(ValueError, match="sidecar"):
        store.commit_receipt(
            cell=_cell("bad-reset"),
            objects=[binding],
            expected_rows=2,
            observed_rows=2,
            exception_rows=0,
            missing_rows=0,
            conservation_failures=0,
            source_snapshot_sha256=SOURCE_SHA,
            runtime_fingerprint={},
            reset_rows=1,
            reset_sidecar_rows=0,
        )


def test_live_object_corruption_invalidates_receipt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    receipt = _receipt(store, cell, b"rows")
    object_path = tmp_path / receipt["objects"][0]["path"]
    object_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="live bytes"):
        store.verify_receipt(store.receipt_path("cell0"), expected_cell=cell)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("source", "receipt lineage mismatch"),
        ("namespace", "runtime fingerprint lineage mismatch"),
        ("runner", "runtime fingerprint lineage mismatch"),
    ),
)
def test_coordinated_receipt_lineage_rehash_is_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    _receipt(store, cell, b"rows")
    path = store.receipt_path("cell0")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "source":
        receipt["source_snapshot_sha256"] = "4" * 64
    elif mutation == "namespace":
        receipt["runtime_fingerprint"]["seed_namespace"] = "formal"
    else:
        receipt["runtime_fingerprint"]["runner_id"] = "other_runner"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha(receipt)
    path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=message):
        store.verify_receipt(path, expected_cell=cell)


@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_receipt_requires_exact_top_level_schema_after_rehash(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    _receipt(store, cell, b"rows")
    path = store.receipt_path("cell0")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    if mutation == "extra":
        receipt["unregistered"] = True
    else:
        receipt.pop("source_snapshot_sha256")
    receipt["receipt_sha256"] = _sha(receipt)
    path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="top-level schema drift"):
        store.verify_receipt(path, expected_cell=cell)


def test_receipt_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    _receipt(store, cell, b"rows")
    path = store.receipt_path("cell0")
    payload = path.read_text(encoding="utf-8")
    path.write_text(
        payload.replace(
            '"task_id":"T-RISK-20260728-04"',
            '"task_id":"WRONG","task_id":"T-RISK-20260728-04"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        store.verify_receipt(path, expected_cell=cell)


@pytest.mark.parametrize(
    ("source", "fingerprint"),
    (
        ("4" * 64, _fingerprint()),
        (SOURCE_SHA, _fingerprint(seed_namespace="formal")),
        (SOURCE_SHA, _fingerprint(runner_id="other_runner")),
    ),
)
def test_commit_rejects_source_namespace_or_runner_outside_store_lineage(
    tmp_path: Path,
    source: str,
    fingerprint: dict[str, object],
) -> None:
    store = _store(tmp_path)
    binding = store.put_bytes(b"rows", role="round_ledger_csv")
    with pytest.raises(ValueError, match="store lineage"):
        store.commit_receipt(
            cell=_cell("cell0"),
            objects=[binding],
            expected_rows=2,
            observed_rows=2,
            exception_rows=0,
            missing_rows=0,
            conservation_failures=0,
            source_snapshot_sha256=source,
            runtime_fingerprint=fingerprint,
            reset_rows=0,
            reset_sidecar_rows=0,
        )
    assert not store.receipt_path("cell0").exists()


@pytest.mark.parametrize(
    "fingerprint",
    (
        _fingerprint(hidden=True),
        {
            key: value
            for key, value in _fingerprint().items()
            if key != "numpy"
        },
        _fingerprint(
            thread_environment={
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        ),
        {"runner_id": RUNNER_ID, "seed_namespace": SEED_NAMESPACE},
    ),
)
def test_commit_rejects_noncanonical_runtime_fingerprint_before_publication(
    tmp_path: Path,
    fingerprint: dict[str, object],
) -> None:
    store = _store(tmp_path)
    binding = store.put_bytes(b"rows", role="round_ledger_csv")
    with pytest.raises(ValueError, match="runtime fingerprint"):
        store.commit_receipt(
            cell=_cell("cell0"),
            objects=[binding],
            expected_rows=2,
            observed_rows=2,
            exception_rows=0,
            missing_rows=0,
            conservation_failures=0,
            source_snapshot_sha256=SOURCE_SHA,
            runtime_fingerprint=fingerprint,
            reset_rows=0,
            reset_sidecar_rows=0,
        )
    assert not store.receipt_path("cell0").exists()


def test_coordinated_nested_runtime_extra_rehash_is_rejected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    _receipt(store, cell, b"rows")
    path = store.receipt_path("cell0")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["runtime_fingerprint"]["hidden"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha(receipt)
    path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="runtime fingerprint"):
        store.verify_receipt(path, expected_cell=cell)


def test_receipt_cell_identity_rejects_bool_integer_type_alias(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    _receipt(store, cell, b"rows")
    path = store.receipt_path("cell0")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["cell"]["plan_index"] = False
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha(receipt)
    path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cell identity"):
        store.verify_receipt(path, expected_cell=cell)


def test_store_rejects_zero_lineage_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonzero lowercase"):
        ImmutableObjectStore(
            repository_root=tmp_path,
            object_root=tmp_path / "objects",
            staging_root=tmp_path / "staging",
            receipt_root=tmp_path / "receipts",
            task_id="T-RISK-20260728-04",
            run_id="fixture",
            config_sha256="0" * 64,
            plan_sha256="2" * 64,
            source_snapshot_sha256="3" * 64,
            seed_namespace=SEED_NAMESPACE,
            runner_id=RUNNER_ID,
        )


def test_unknown_extra_and_missing_receipts_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _receipt(store, _cell("cell0", 0), b"zero")
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        store.inventory([_cell("cell0", 0), _cell("cell1", 1)])
    _receipt(store, _cell("extra", 2), b"extra")
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        store.inventory([_cell("cell0", 0)])


def test_complete_inventory_references_objects_without_zip_or_merged_csv(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    cells = [_cell("cell0", 0), _cell("cell1", 1)]
    _receipt(store, cells[0], b"zero")
    _receipt(store, cells[1], b"one")
    inventory = store.inventory(cells)
    assert inventory["receipt_count"] == 2
    assert inventory["unique_object_count"] == 2
    assert inventory["totals"]["expected_rows"] == 4
    assert inventory["totals"]["observed_rows"] == 4
    assert inventory["raw_status"] == "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT"
    assert inventory["scientific_verdict"] is None
    assert inventory["qualified_claim"] is None
    assert inventory["monolithic_archive"] is None
    assert inventory["merged_full_csv"] is None


def test_exception_rows_are_retained_but_terminal_is_incomplete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    binding = store.put_bytes(b"explicit exception row", role="round_ledger_csv")
    store.commit_receipt(
        cell=cell,
        objects=[binding],
        expected_rows=2,
        observed_rows=2,
        exception_rows=1,
        missing_rows=0,
        conservation_failures=0,
        source_snapshot_sha256=SOURCE_SHA,
        runtime_fingerprint=_fingerprint(),
        reset_rows=0,
        reset_sidecar_rows=0,
    )
    inventory = store.inventory([cell])
    assert inventory["totals"]["exception_rows"] == 1
    assert inventory["raw_status"] == "INCOMPLETE_FAIL_CLOSED"


def test_receipt_path_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        store.receipt_path("../escape")
    staged = store.new_staging_path()
    staged.unlink()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"x")
    try:
        staged.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="non-symlink"):
        store.adopt_staged_file(staged, role="bad", media_type="application/octet-stream")


def test_attempt_ledger_detects_hash_chain_mutation(tmp_path: Path) -> None:
    path = tmp_path / "run/attempts.jsonl"
    first = append_attempt_event(
        path,
        task_id="T-RISK-20260728-04",
        run_id="fresh-test-run",
        event="START",
        payload={"formal_outcome_accessed": False},
    )
    second = append_attempt_event(
        path,
        task_id="T-RISK-20260728-04",
        run_id="fresh-test-run",
        event="OBJECT_COMMITTED",
        payload={"chunk_id": "cell0"},
    )
    assert first["sequence"] == 0
    assert second["sequence"] == 1
    assert second["previous_event_sha256"] == first["event_sha256"]
    records = path.read_text(encoding="utf-8").splitlines()
    changed = json.loads(records[0])
    changed["event"] = "MUTATED"
    records[0] = json.dumps(changed, separators=(",", ":"))
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash drift"):
        append_attempt_event(
            path,
            task_id="T-RISK-20260728-04",
            run_id="fresh-test-run",
            event="SHOULD_FAIL",
            payload={},
        )


def test_publish_manifest_keeps_all_claims_null(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cell = _cell("cell0")
    _receipt(store, cell, b"rows")
    inventory = store.inventory([cell])
    inventory_path = tmp_path / "evidence/inventory.json"
    manifest_path = tmp_path / "evidence/manifest.json"
    _, manifest = publish_inventory_and_manifest(
        repository_root=tmp_path,
        inventory_path=inventory_path,
        manifest_path=manifest_path,
        inventory=inventory,
        claim_fields=("twin_qualification", "external_sota", "hardware_measured"),
    )
    assert manifest["scientific_verdict"] is None
    assert manifest["qualified_claim"] is None
    assert set(manifest["claim_boundary"].values()) == {None}
    assert manifest["monolithic_archive"] is None
    assert manifest["merged_full_csv"] is None
    assert inventory_path.exists() and manifest_path.exists()
    assert manifest["inventory"]["path"] == "evidence/inventory.json"
    mutated = dict(inventory)
    mutated["raw_status"] = "MUTATED"
    # Preserve the old self hash deliberately: either self-hash or immutable
    # publication must reject this attempted post-finalize rewrite.
    with pytest.raises((RuntimeError, ValueError)):
        publish_inventory_and_manifest(
            repository_root=tmp_path,
            inventory_path=inventory_path,
            manifest_path=manifest_path,
            inventory=mutated,
            claim_fields=("twin_qualification",),
        )
