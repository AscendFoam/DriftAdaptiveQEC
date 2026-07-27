from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_high_cutoff_design_pilot as subject
from cnn_fpga.benchmark import phase9_fresh_twin_qualification as runner


ROOT = Path(__file__).resolve().parents[1]


def test_live_config_has_bound_sources_and_disjoint_pilot_matrix() -> None:
    pilot, base = subject.load_pilot_config(ROOT)
    execution = subject.materialize_execution_config(pilot, base)
    cells = subject.build_pilot_cells(pilot, execution)

    assert len(cells) == 32
    assert {cell.cutoff for cell in cells} == {16, 20, 24, 28}
    assert {cell.backend for cell in cells} == {"A", "B"}
    assert {cell.scenario for cell in cells} == {
        "step",
        "telegraph",
        "burst",
        "compound",
    }
    assert all(cell.sample_count == 72 for cell in cells)
    assert all(cell.expected_rows == 864 for cell in cells)
    assert sum(cell.expected_rows for cell in cells) == 27648

    first_a = next(cell for cell in cells if cell.backend == "A")
    first_b = next(cell for cell in cells if cell.backend == "B")
    assert runner._seed_for(execution, first_a, 0) == 1410000
    assert runner._seed_for(execution, first_b, 0) == 1411000
    assert execution["formal_splits"]["heldout_common"]["start"] == 1412000
    assert pilot["claim_boundary"]["twin_qualification"] is None
    assert pilot["optional_cutoff_32"]["enabled"] is False
    assert pilot["diagnostic_contract"]["multiplier_replicates"] == 199
    assert pilot["diagnostic_contract"]["multiplier_seed_namespace"] == 1420000
    assert pilot["diagnostic_contract"]["formal_rescue_forbidden"] is True
    for partition in pilot["stage_partition"].values():
        rounds = [
            round_index for indices in partition.values() for round_index in indices
        ]
        assert sorted(rounds) == list(range(12))
        assert len(rounds) == len(set(rounds)) == 12


def test_materialization_does_not_mutate_bound_base_config() -> None:
    pilot, base = subject.load_pilot_config(ROOT)
    before = json.dumps(base, sort_keys=True)
    execution = subject.materialize_execution_config(pilot, base)
    assert json.dumps(base, sort_keys=True) == before
    assert execution is not base
    assert execution["formal_matrix"]["trajectory_sample_count"] == 72


def test_pilot_execution_is_blocked_while_hardened_confirmation_is_pending() -> None:
    with pytest.raises(RuntimeError, match="pending and unreleased"):
        subject.load_pilot_config(ROOT, require_hardened=True)


def test_receipt_analysis_and_byte_bindings_fail_closed(tmp_path, monkeypatch) -> None:
    pilot = {
        "artifact_paths": {"receipt_directory": "receipts"},
        "purpose": "fixture",
    }
    cell = runner.CellSpec(
        chunk_id="chunk",
        layer="fault",
        cell_base="fault|step",
        cutoff=16,
        backend="A",
        sample_count=6,
        convergence_role="pilot",
        scenario="step",
        horizon=12,
    )
    csv_path = tmp_path / "chunk.csv"
    npz_path = tmp_path / "chunk.npz"
    csv_path.write_bytes(b"csv")
    npz_path.write_bytes(b"npz")
    monkeypatch.setattr(runner, "_validate_chunk_files", lambda *_args: None)
    run_identity = {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "analysis_sha256": "identity-hash",
    }
    execution_analysis_sha256 = "execution-hash"
    receipt = {
        "task_id": subject.TASK_ID,
        "schema_version": subject.RECEIPT_SCHEMA,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "config_analysis_sha256": subject._sha(pilot),
        "execution_analysis_sha256": execution_analysis_sha256,
        "pilot_source_sha256": sha256(Path(subject.__file__).read_bytes()).hexdigest(),
        "cell": asdict(cell),
        "chunk_id": cell.chunk_id,
        "cell_base": cell.cell_base,
        "layer": cell.layer,
        "backend": cell.backend,
        "cutoff": cell.cutoff,
        "expected_rows": cell.expected_rows,
        "observed_rows": cell.expected_rows,
        "exception_rows": 0,
        "csv": subject._binding(csv_path, tmp_path),
        "npz": subject._binding(npz_path, tmp_path),
    }
    receipt["analysis_sha256"] = subject._sha(receipt)
    subject._validate_receipt(
        tmp_path,
        pilot,
        cell,
        receipt,
        run_identity=run_identity,
        execution_analysis_sha256=execution_analysis_sha256,
    )

    csv_path.write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="csv binding drift"):
        subject._validate_receipt(
            tmp_path,
            pilot,
            cell,
            receipt,
            run_identity=run_identity,
            execution_analysis_sha256=execution_analysis_sha256,
        )


def test_owner_lock_rejects_second_supervisor_and_cleans_up(tmp_path) -> None:
    pilot = {"artifact_paths": {"owner_lock": "run/supervisor.owner.lock"}}
    lock_path = tmp_path / "run" / "supervisor.owner.lock"
    with subject._exclusive_owner_lock(tmp_path, pilot):
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="owner lock already exists"):
            with subject._exclusive_owner_lock(tmp_path, pilot):
                pass
    assert not lock_path.exists()


def test_run_identity_rejects_synchronized_tamper(tmp_path, monkeypatch) -> None:
    pilot = {
        "artifact_paths": {"run_identity": "run/run_identity.json"},
        "purpose": "fixture",
    }
    execution_hash = "execution-hash"
    identity = subject._load_or_create_run_identity(tmp_path, pilot, execution_hash)
    assert identity["run_id"]
    path = tmp_path / "run" / "run_identity.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["config_analysis_sha256"] = "attacker-rehash"
    unsigned = dict(tampered)
    unsigned.pop("analysis_sha256")
    tampered["analysis_sha256"] = subject._sha(unsigned)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="run identity binding drift"):
        subject._load_or_create_run_identity(tmp_path, pilot, execution_hash)


def test_manifest_rejects_rehashed_claim_tamper(tmp_path, monkeypatch) -> None:
    pilot = {
        "base_config": {"path": "base.json"},
        "source_bindings": {},
        "artifact_paths": {"run_identity": "run_identity.json"},
        "hardened_confirmation_source": {
            "report": {"path": "hardened.json"},
            "source_data": {"path": "hardened.csv"},
        },
        "claim_boundary": dict(subject.CLAIM_BOUNDARY),
    }
    execution = {"fixture": True}
    cell = runner.CellSpec(
        chunk_id="chunk",
        layer="fault",
        cell_base="fault|step",
        cutoff=16,
        backend="A",
        sample_count=1,
        convergence_role="pilot",
        scenario="step",
        horizon=1,
    )
    run_identity = {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "analysis_sha256": "identity-hash",
    }
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda path, root: {
            "path": Path(path).name,
            "bytes": 1,
            "sha256": "bound",
        },
    )
    monkeypatch.setattr(subject, "_validate_receipt_file", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_chunk_health", lambda *_a, **_k: (0, 0))
    bindings = {
        "config": subject._binding(tmp_path / subject.CONFIG_PATH, tmp_path),
        "base_config": subject._binding(tmp_path / "base.json", tmp_path),
        "pilot_source": subject._binding(Path(subject.__file__), tmp_path),
        "run_identity": subject._binding(tmp_path / "run_identity.json", tmp_path),
        "hardened_confirmation_report": subject._binding(
            tmp_path / "hardened.json", tmp_path
        ),
        "hardened_confirmation_source_data": subject._binding(
            tmp_path / "hardened.csv", tmp_path
        ),
    }
    manifest = {
        "task_id": subject.TASK_ID,
        "schema_version": subject.MANIFEST_SCHEMA,
        "status": subject.STATUS,
        "scientific_verdict": None,
        "qualified_claim": None,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "config_analysis_sha256": subject._sha(pilot),
        "execution_analysis_sha256": subject._sha(execution),
        "pilot_source_sha256": sha256(Path(subject.__file__).read_bytes()).hexdigest(),
        "observed_cells": 1,
        "observed_rows": cell.expected_rows,
        "exception_rows": 0,
        "conservation_failure_rows": 0,
        "chunk_receipts": [{"cell": {"chunk_id": cell.chunk_id}}],
        "receipt_bindings": [{"path": "receipt.json"}],
        "claim_state": dict(subject.CLAIM_BOUNDARY),
        "bindings": bindings,
        "runtime": {},
    }
    manifest["analysis_sha256"] = subject._sha(manifest)
    subject._verify_manifest(
        tmp_path,
        pilot,
        execution,
        [cell],
        run_identity,
        manifest,
    )

    tampered = dict(manifest)
    tampered["qualified_claim"] = "attacker-upgrade"
    tampered.pop("analysis_sha256")
    tampered["analysis_sha256"] = subject._sha(tampered)
    with pytest.raises(RuntimeError, match="semantic drift"):
        subject._verify_manifest(
            tmp_path,
            pilot,
            execution,
            [cell],
            run_identity,
            tampered,
        )


def test_help_exits_before_runner(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(subject, "run_pilot", lambda *_args: calls.append(True))
    with pytest.raises(SystemExit) as raised:
        subject.main(["--help"])
    assert raised.value.code == 0
    assert calls == []


def _install_run_pilot_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    duplicate_receipt_set: bool = False,
) -> tuple[list[runner.CellSpec], Path, Path]:
    pilot = {
        "max_workers": 2,
        "base_config": {"path": "base.json"},
        "source_bindings": {},
        "artifact_paths": {
            "execution_manifest": "run/manifest.json",
            "heartbeat": "run/heartbeat.json",
            "receipt_directory": "run/receipts",
            "run_identity": "run/run_identity.json",
        },
        "hardened_confirmation_source": {
            "report": {"path": "hardened.json"},
            "source_data": {"path": "hardened.csv"},
        },
        "claim_boundary": dict(subject.CLAIM_BOUNDARY),
    }
    execution = {"fixture": True}
    cells = [
        runner.CellSpec(
            chunk_id=f"chunk-{backend}",
            layer="fault",
            cell_base=f"fault|step|{backend}",
            cutoff=16,
            backend=backend,
            sample_count=1,
            convergence_role="pilot",
            scenario="step",
            horizon=1,
        )
        for backend in ("A", "B")
    ]
    run_identity = {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "analysis_sha256": "identity-hash",
    }

    def receipt_for(cell: runner.CellSpec) -> dict:
        selected = cells[0] if duplicate_receipt_set else cell
        return {
            "cell": asdict(selected),
            "csv": {"path": f"run/{selected.chunk_id}.csv"},
            "npz": {"path": f"run/{selected.chunk_id}.npz"},
        }

    class DoneFuture:
        def __init__(self, receipt: dict) -> None:
            self._receipt = receipt

        def result(self) -> dict:
            return self._receipt

    class ImmediatePool:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def submit(self, _function, *args):
            cell = runner.CellSpec(**args[3])
            return DoneFuture(receipt_for(cell))

    monkeypatch.setattr(
        subject, "load_pilot_config", lambda *_a, **_k: (pilot, {"base": True})
    )
    monkeypatch.setattr(
        subject, "materialize_execution_config", lambda *_a, **_k: execution
    )
    monkeypatch.setattr(subject, "build_pilot_cells", lambda *_a, **_k: cells)
    monkeypatch.setattr(
        subject, "_exclusive_owner_lock", lambda *_a, **_k: nullcontext()
    )
    monkeypatch.setattr(
        subject, "_load_or_create_run_identity", lambda *_a, **_k: run_identity
    )
    monkeypatch.setattr(subject, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(subject, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda path, _root: {
            "path": Path(path).name,
            "bytes": 1,
            "sha256": "bound",
        },
    )
    monkeypatch.setattr(subject, "_validate_receipt_file", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_chunk_health", lambda *_a, **_k: (0, 0))
    return (
        cells,
        tmp_path / pilot["artifact_paths"]["execution_manifest"],
        tmp_path / pilot["artifact_paths"]["heartbeat"],
    )


def _assert_failed_heartbeat(
    heartbeat_path: Path,
    *,
    error_type: str,
) -> None:
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["active"] is False
    assert heartbeat["state"] == "FAILED"
    assert heartbeat["error_type"] == error_type
    assert heartbeat["analysis_sha256"] == subject._sha(
        {key: value for key, value in heartbeat.items() if key != "analysis_sha256"}
    )


def test_receipt_set_drift_finalization_fails_closed(tmp_path, monkeypatch) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path,
        monkeypatch,
        duplicate_receipt_set=True,
    )

    with pytest.raises(RuntimeError, match="receipt cell set drift"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="RuntimeError")
    assert not manifest_path.exists()


def test_chunk_health_exception_finalization_fails_closed(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        subject,
        "_chunk_health",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("health fault")),
    )

    with pytest.raises(OSError, match="health fault"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="OSError")
    assert not manifest_path.exists()


def test_manifest_write_exception_finalization_fails_closed(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    atomic_text = subject._atomic_text

    def fail_manifest_write(path: Path, text: str) -> None:
        if Path(path) == manifest_path:
            raise PermissionError("manifest write fault")
        atomic_text(path, text)

    monkeypatch.setattr(subject, "_atomic_text", fail_manifest_write)

    with pytest.raises(PermissionError, match="manifest write fault"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="PermissionError")
    assert not manifest_path.exists()


def test_manifest_live_verify_exception_removes_pseudo_complete(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    verify_manifest = subject._verify_manifest
    calls = 0

    def fail_live_verify(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("live manifest verify fault")
        verify_manifest(*args, **kwargs)

    monkeypatch.setattr(subject, "_verify_manifest", fail_live_verify)

    with pytest.raises(RuntimeError, match="live manifest verify fault"):
        subject.run_pilot(tmp_path)

    assert calls == 2
    _assert_failed_heartbeat(heartbeat_path, error_type="RuntimeError")
    assert not manifest_path.exists()


def test_complete_heartbeat_exception_finalization_fails_closed(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    heartbeat = subject._heartbeat

    def fail_complete_heartbeat(*args, **kwargs) -> None:
        if kwargs.get("state") == "COMPLETE":
            raise OSError("complete heartbeat fault")
        heartbeat(*args, **kwargs)

    monkeypatch.setattr(subject, "_heartbeat", fail_complete_heartbeat)

    with pytest.raises(OSError, match="complete heartbeat fault"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="OSError")
    assert not manifest_path.exists()


def test_healthy_finalization_commits_manifest_before_complete(
    tmp_path, monkeypatch
) -> None:
    cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )

    report = subject.run_pilot(tmp_path)

    assert report == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["status"] == subject.STATUS
    assert report["observed_cells"] == len(cells)
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["completed_cells"] == len(cells)
    assert heartbeat["active"] is False
    assert heartbeat["state"] == "COMPLETE"
    assert heartbeat["error_type"] is None
