from __future__ import annotations

import csv
from hashlib import sha256
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_high_cutoff_design_bootstrap_v2 as bootstrap
from cnn_fpga.benchmark import phase9_high_cutoff_design_diagnostic as diagnostic
from cnn_fpga.benchmark import phase9_high_cutoff_design_pilot as pilot


def _chunk_fixture(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    *,
    nonterminal_certificate: str = "",
    terminal_certificate: str = "1e-7",
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    chunk_id = "fixture"
    monkeypatch.setattr(diagnostic, "pilot_runner", pilot)
    receipt: dict[str, object] = {
        "task_id": diagnostic.TASK_ID,
        "schema_version": pilot.RECEIPT_SCHEMA,
        "run_id": "run",
        "run_identity_analysis_sha256": "identity",
        "config_analysis_sha256": "config",
        "execution_analysis_sha256": "execution",
        "input_snapshot_analysis_sha256": "snapshot",
        "pilot_source_sha256": "pilot",
        "chunk_id": chunk_id,
        "cell": {"chunk_id": chunk_id},
        "csv": {"kind": "csv"},
        "npz": {"kind": "npz"},
    }
    manifest = {
        "run_id": "run",
        "run_identity_analysis_sha256": "identity",
        "config_analysis_sha256": "config",
        "execution_analysis_sha256": "execution",
        "input_snapshot_analysis_sha256": "snapshot",
        "pilot_source_sha256": "pilot",
    }
    config = {"artifact_paths": {"receipt_directory": "receipts"}}
    fields = [
        "row_id",
        "exception_type",
        "conservation_pass",
        "cutoff",
        "seed_position",
        "round_index",
        "terminal_round",
        "mean_photon",
        "level_g",
        "level_e",
        "level_f",
        "logical_survival",
        "density_quantization_trace_distance_bound",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerow(
        {
            "row_id": "nonterminal",
            "exception_type": "",
            "conservation_pass": "True",
            "cutoff": "1",
            "seed_position": "0",
            "round_index": "0",
            "terminal_round": "False",
            "mean_photon": "0",
            "level_g": "1",
            "level_e": "0",
            "level_f": "0",
            "logical_survival": "1",
            "density_quantization_trace_distance_bound": nonterminal_certificate,
        }
    )
    writer.writerow(
        {
            "row_id": "terminal",
            "exception_type": "",
            "conservation_pass": "True",
            "cutoff": "1",
            "seed_position": "0",
            "round_index": "1",
            "terminal_round": "True",
            "mean_photon": "0",
            "level_g": "1",
            "level_e": "0",
            "level_f": "0",
            "logical_survival": "1",
            "density_quantization_trace_distance_bound": terminal_certificate,
        }
    )
    csv_bytes = stream.getvalue().encode()
    archive = io.BytesIO()
    np.savez(
        archive,
        density_row_ids=np.asarray(["terminal"]),
        densities=np.asarray([np.eye(3) / 3], dtype=np.complex64),
    )

    monkeypatch.setattr(diagnostic, "_verify_self_hash", lambda *_args: None)
    monkeypatch.setattr(
        diagnostic.pilot_runner,
        "_read_bound_json",
        lambda *_args, **_kwargs: (
            (tmp_path / "receipts" / f"{chunk_id}.json").resolve(),
            receipt,
        ),
    )

    def read_bytes(_root: Path, binding: dict[str, str]):
        return (
            tmp_path,
            csv_bytes if binding["kind"] == "csv" else archive.getvalue(),
        )

    monkeypatch.setattr(diagnostic.pilot_runner, "_read_bound_bytes", read_bytes)
    original_parser = diagnostic._parse_chunk
    bootstrap._install_terminal_quantization_parser(diagnostic)
    request.addfinalizer(
        lambda: setattr(diagnostic, "_parse_chunk", original_parser)
    )
    return receipt, config, manifest


def test_v2_external_launcher_is_self_consistent() -> None:
    assert (
        sha256(bootstrap.EXTERNAL_LAUNCHER_SOURCE.encode()).hexdigest()
        == bootstrap.EXTERNAL_LAUNCHER_SHA256
    )
    assert (
        "phase9_high_cutoff_design_bootstrap_v2.py"
        in bootstrap.EXTERNAL_LAUNCHER_SOURCE
    )


def test_v2_explicitly_rebinds_diagnostic_launcher_contract() -> None:
    module = SimpleNamespace(EXTERNAL_LAUNCHER_SHA256="legacy")
    bootstrap._bind_v2_diagnostic_launcher_contract(module)
    assert module.EXTERNAL_LAUNCHER_SHA256 == bootstrap.EXTERNAL_LAUNCHER_SHA256


def test_v2_accepts_terminal_only_quantization_certificate(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    receipt, config, manifest = _chunk_fixture(
        monkeypatch, request, tmp_path
    )
    rows, densities = diagnostic._parse_chunk(
        tmp_path,
        receipt,
        {"path": "receipt"},
        config=config,
        manifest=manifest,
    )
    assert rows[0]["density_quantization_trace_distance_bound"] is None
    assert rows[1]["density_quantization_trace_distance_bound"] == pytest.approx(
        1e-7
    )
    assert set(densities) == {"terminal"}


@pytest.mark.parametrize(
    ("nonterminal", "terminal", "message"),
    [
        ("1e-7", "1e-7", "non-terminal density quantization certificate present"),
        ("", "", "terminal density quantization certificate missing"),
        ("", "nan", "terminal density quantization certificate invalid"),
        ("", "-1e-7", "terminal density quantization certificate invalid"),
    ],
)
def test_v2_rejects_quantization_certificate_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    nonterminal: str,
    terminal: str,
    message: str,
) -> None:
    receipt, config, manifest = _chunk_fixture(
        monkeypatch,
        request,
        tmp_path,
        nonterminal_certificate=nonterminal,
        terminal_certificate=terminal,
    )
    with pytest.raises(ValueError, match=message):
        diagnostic._parse_chunk(
            tmp_path,
            receipt,
            {"path": "receipt"},
            config=config,
            manifest=manifest,
        )
