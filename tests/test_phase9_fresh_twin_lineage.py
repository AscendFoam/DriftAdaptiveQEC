from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from cnn_fpga.benchmark.phase9_fresh_twin_lineage import (
    HISTORICAL_BINDINGS,
    HISTORICAL_NULL_FIELDS,
    PASS_VERDICT,
    PROHIBITED_HISTORICAL_BASENAMES,
    build_receipt,
    verify_receipt,
    write_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def _historical_root(tmp_path: Path) -> Path:
    for binding in HISTORICAL_BINDINGS.values():
        relative = Path(binding["path"])
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def test_live_historical_no_go_is_byte_bound_and_null() -> None:
    report = build_receipt(ROOT)
    assert report["verdict"] == PASS_VERDICT
    assert report["gate_summary"] == {"passed": 10, "total": 10}
    assert report["historical_parent_rewritten"] is False
    assert report["fresh_source_scan"]["violations"] == []
    for binding in report["historical_bindings"].values():
        assert binding["claim_fields"] == list(HISTORICAL_NULL_FIELDS)
        assert binding["all_claims_literal_null"] is True


@pytest.mark.parametrize("name", sorted(HISTORICAL_BINDINGS))
def test_each_historical_byte_mutation_is_rejected(
    tmp_path: Path, name: str
) -> None:
    root = _historical_root(tmp_path)
    path = root / HISTORICAL_BINDINGS[name]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte hash mismatch"):
        build_receipt(root)


def test_typed_string_null_cannot_replace_literal_null(tmp_path: Path) -> None:
    root = _historical_root(tmp_path)
    name = "formal_report"
    path = root / HISTORICAL_BINDINGS[name]["path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["claim_state"][HISTORICAL_NULL_FIELDS[0]] = "null"
    path.write_text(json.dumps(document), encoding="utf-8")
    # Byte pin fails first and therefore cannot be bypassed by semantic edits.
    with pytest.raises(ValueError, match="byte hash mismatch"):
        build_receipt(root)


def test_fresh_source_cannot_reference_historical_cell_artifact(
    tmp_path: Path,
) -> None:
    root = _historical_root(tmp_path)
    source = root / "cnn_fpga/benchmark/phase9_fresh_twin_bad.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        f'BAD = "{PROHIBITED_HISTORICAL_BASENAMES[0]}"\n', encoding="utf-8"
    )
    report = build_receipt(root)
    assert report["verdict"] == "INCOMPLETE_FAIL_CLOSED"
    assert report["gates"]["G10_no_fresh_cell_level_historical_access"] is False


def test_unrelated_sources_are_not_scanned(tmp_path: Path) -> None:
    root = _historical_root(tmp_path)
    source = root / "scratch.py"
    source.write_text(PROHIBITED_HISTORICAL_BASENAMES[0], encoding="utf-8")
    assert build_receipt(root)["verdict"] == PASS_VERDICT


def test_receipt_verification_rejects_arbitrary_nonzero_analysis() -> None:
    report = build_receipt(ROOT)
    forged = copy.deepcopy(report)
    forged["analysis_sha256"] = "f" * 64
    assert not verify_receipt(forged, ROOT)


def test_receipt_write_is_deterministic(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    first = write_receipt(destination, ROOT)
    first_bytes = destination.read_bytes()
    second = write_receipt(destination, ROOT)
    assert first == second
    assert destination.read_bytes() == first_bytes


def test_cli_override_is_rejected() -> None:
    from cnn_fpga.benchmark.phase9_fresh_twin_lineage import main

    with pytest.raises(SystemExit) as raised:
        main(["--root", "elsewhere"])
    assert raised.value.code == 2
