"""Stdlib-only process entry seal for T04 numerical commands.

Thread-control variables are checked before importing NumPy/SciPy or any
physics module.  This prevents a configuration that merely *declares*
single-threaded BLAS while the already-imported runtime remains multithreaded.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Iterable


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
FORBIDDEN_PREIMPORT_PREFIXES = ("numpy", "scipy")
DISPATCH = {
    "contract": "cnn_fpga.benchmark.phase9_powered_twin_plan",
    "preformal": "cnn_fpga.benchmark.phase9_powered_twin_preformal_audit",
    "resource": "cnn_fpga.benchmark.phase9_powered_twin_preflight",
    "production": "cnn_fpga.benchmark.phase9_powered_twin_qualification",
    "verify": "cnn_fpga.benchmark.phase9_powered_twin_verifier",
}


def assert_process_entry_seal() -> None:
    drift = {
        name: os.environ.get(name)
        for name, expected in THREAD_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if drift:
        raise RuntimeError(
            "T04 process-entry thread environment drift: "
            + ",".join(f"{key}={value!r}" for key, value in sorted(drift.items()))
        )
    imported = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FORBIDDEN_PREIMPORT_PREFIXES
        )
    )
    if imported:
        raise RuntimeError(
            "T04 numerical dependency imported before process-entry seal: "
            + ",".join(imported[:10])
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one T04 command behind the pre-import runtime seal."
    )
    parser.add_argument("mode", choices=tuple(DISPATCH))
    arguments, remainder = parser.parse_known_args(
        list(argv) if argv is not None else None
    )
    assert_process_entry_seal()
    module = importlib.import_module(DISPATCH[arguments.mode])
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise RuntimeError(f"{DISPATCH[arguments.mode]} has no callable main")
    return int(entry(remainder))


if __name__ == "__main__":
    raise SystemExit(main())

