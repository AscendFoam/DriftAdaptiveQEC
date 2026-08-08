"""Fail-closed CLI contract for fresh-twin artifact writers.

The module entry points have no scientific overrides.  Help and invalid
arguments must therefore terminate before any canonical artifact writer is
called, including when ``main()`` receives no explicit argv and argparse must
consume ``sys.argv``.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pytest

from cnn_fpga.benchmark import phase9_fresh_twin_design_power as design_power
from cnn_fpga.benchmark import phase9_fresh_twin_lineage as lineage
from cnn_fpga.benchmark import phase9_fresh_twin_readout_power as readout_power
from cnn_fpga.benchmark import phase9_fresh_twin_verifier as verifier


MODULE_WRITERS = (
    (lineage, "write_receipt"),
    (design_power, "write_artifacts"),
    (readout_power, "write_artifacts"),
    (verifier, "write_artifacts"),
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUTPUTS = (
    ROOT / "docs/t_risk_20260726_01_historical_no_go_receipt.json",
    ROOT / "docs/t_risk_20260726_01_design_power.json",
    ROOT / "docs/t_risk_20260726_01_design_power_source_data.csv",
    ROOT / "docs/t_risk_20260726_01_readout_power.json",
    ROOT / "docs/t_risk_20260726_01_readout_power_source_data.csv",
    ROOT / verifier.DEFAULT_REPORT_PATH,
    ROOT / verifier.DEFAULT_QUALIFICATION_PATH,
    ROOT / verifier.DEFAULT_SOURCE_PATH,
    ROOT / verifier.DEFAULT_GATE_LEDGER_PATH,
    ROOT / verifier.DEFAULT_RELEASE_PATH,
    ROOT / verifier.DEFAULT_RELEASE_PIN_PATH,
)


def _filesystem_snapshot() -> dict[str, tuple[bool, int, int, str | None]]:
    snapshot: dict[str, tuple[bool, int, int, str | None]] = {}
    for path in CANONICAL_OUTPUTS:
        if path.exists():
            stat = path.stat()
            snapshot[path.as_posix()] = (
                True,
                stat.st_size,
                stat.st_mtime_ns,
                sha256(path.read_bytes()).hexdigest(),
            )
        else:
            snapshot[path.as_posix()] = (False, 0, 0, None)
    return snapshot


@pytest.mark.parametrize(("module", "writer_name"), MODULE_WRITERS)
def test_help_is_zero_write(module, writer_name, monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(module, writer_name, lambda *args, **kwargs: calls.append(args))

    with pytest.raises(SystemExit) as raised:
        module.main(["--help"])

    assert raised.value.code == 0
    assert calls == []


@pytest.mark.parametrize(("module", "writer_name"), MODULE_WRITERS)
def test_unknown_argument_is_zero_write_when_main_consumes_sys_argv(
    module, writer_name, monkeypatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(module, writer_name, lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(module.__file__), "--not-a-valid-fresh-twin-option"],
    )

    with pytest.raises(SystemExit) as raised:
        module.main()

    assert raised.value.code == 2
    assert calls == []


@pytest.mark.parametrize(("module", "_writer_name"), MODULE_WRITERS)
def test_help_and_unknown_arguments_preserve_live_artifact_hashes_and_mtimes(
    module, _writer_name
) -> None:
    before = _filesystem_snapshot()
    with pytest.raises(SystemExit) as help_exit:
        module.main(["--help"])
    assert help_exit.value.code == 0
    assert _filesystem_snapshot() == before

    with pytest.raises(SystemExit) as invalid_exit:
        module.main(["--not-a-valid-fresh-twin-option"])
    assert invalid_exit.value.code == 2
    assert _filesystem_snapshot() == before
