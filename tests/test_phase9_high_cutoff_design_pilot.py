from __future__ import annotations

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
