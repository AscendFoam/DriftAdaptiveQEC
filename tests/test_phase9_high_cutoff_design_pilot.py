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
    assert runner._seed_for(execution, first_a, 0) == 1310000
    assert runner._seed_for(execution, first_b, 0) == 1311000
    assert execution["formal_splits"]["heldout_common"]["start"] == 1312000
    assert pilot["claim_boundary"]["twin_qualification"] is None
    assert pilot["optional_cutoff_32"]["enabled"] is False
    assert pilot["diagnostic_contract"]["multiplier_replicates"] == 1999
    assert pilot["diagnostic_contract"]["formal_rescue_forbidden"] is True
    for partition in pilot["stage_partition"].values():
        rounds = [
            round_index
            for indices in partition.values()
            for round_index in indices
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
    receipt = {
        "task_id": subject.TASK_ID,
        "schema_version": "fixture",
        "config_analysis_sha256": subject._sha(pilot),
        "cell": asdict(cell),
        "csv": subject._binding(csv_path, tmp_path),
        "npz": subject._binding(npz_path, tmp_path),
    }
    receipt["analysis_sha256"] = subject._sha(receipt)
    subject._validate_receipt(tmp_path, pilot, cell, receipt)

    csv_path.write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="csv binding drift"):
        subject._validate_receipt(tmp_path, pilot, cell, receipt)


def test_help_exits_before_runner(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(subject, "run_pilot", lambda *_args: calls.append(True))
    with pytest.raises(SystemExit) as raised:
        subject.main(["--help"])
    assert raised.value.code == 0
    assert calls == []
