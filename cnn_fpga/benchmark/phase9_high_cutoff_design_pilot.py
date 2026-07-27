"""Resumable state-conditioned high-cutoff design pilot.

The pilot reuses the already validated physics execution kernel but has a new,
disjoint seed namespace and a deliberately small denominator.  It produces
raw chunks only for designing the subsequent formal matrix.  It cannot emit
a twin-qualification verdict or release any blocked downstream task.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

from cnn_fpga.benchmark import phase9_fresh_twin_qualification as runner


TASK_ID = "T-RISK-20260727-01"
CONFIG_PATH = "configs/phase9/t_risk_20260727_01_high_cutoff_design_pilot.json"
CONFIG_SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-PILOT-CONFIG-V1"
MANIFEST_SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-PILOT-MANIFEST-V1"
STATUS = "DESIGN_PILOT_RAW_EVIDENCE_COMPLETE"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_pilot_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("claim_boundary", {}).get("design_pilot_only") is not True
        or any(
            value is not None
            for key, value in config.get("claim_boundary", {}).items()
            if key != "design_pilot_only"
        )
    ):
        raise ValueError("high-cutoff pilot identity/claim firewall invalid")
    if (
        config.get("cutoffs") != [16, 20, 24, 28]
        or int(config.get("trajectory_count", 0))
        != 6 * int(config.get("clusters_per_state", -1))
        or sorted(config.get("scenario_names", []))
        != ["burst", "compound", "step", "telegraph"]
    ):
        raise ValueError("high-cutoff pilot matrix drift")
    for scenario in config["scenario_names"]:
        partition = config.get("stage_partition", {}).get(scenario)
        if (
            not isinstance(partition, Mapping)
            or sorted(
                round_index
                for indices in partition.values()
                for round_index in indices
            )
            != list(range(12))
            or sum(len(indices) for indices in partition.values()) != 12
        ):
            raise ValueError(f"high-cutoff stage partition drift: {scenario}")
    splits = config["seed_splits"]
    intervals = []
    for key in ("trajectory_backend_a", "trajectory_backend_b", "heldout_common"):
        start = int(splits[key]["start"])
        count = int(splits[key]["count"])
        intervals.append(set(range(start, start + count)))
    if (
        splits.get("all_intervals_disjoint") is not True
        or splits.get("disjoint_from_20260726_formal") is not True
        or any(intervals[i] & intervals[j] for i in range(3) for j in range(i))
        or min(min(interval) for interval in intervals) <= 1076767
    ):
        raise ValueError("high-cutoff pilot seed firewall invalid")
    for binding in config["source_bindings"].values():
        path = root / str(binding["path"])
        if _binding(path, root)["sha256"] != binding["sha256"]:
            raise ValueError(f"pilot source binding drift: {binding['path']}")
    base_binding = config["base_config"]
    base_path = root / str(base_binding["path"])
    if _binding(base_path, root)["sha256"] != base_binding["sha256"]:
        raise ValueError("pilot base config binding drift")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    return config, base


def materialize_execution_config(
    pilot: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    execution = json.loads(json.dumps(base))
    count = int(pilot["trajectory_count"])
    execution["formal_matrix"]["trajectory_sample_count"] = count
    execution["formal_matrix"]["cutoff_ladder"] = list(pilot["cutoffs"])
    execution["formal_splits"]["trajectory_backend_a"] = dict(
        pilot["seed_splits"]["trajectory_backend_a"]
    )
    execution["formal_splits"]["trajectory_backend_b"] = dict(
        pilot["seed_splits"]["trajectory_backend_b"]
    )
    execution["formal_splits"]["heldout_common"] = dict(
        pilot["seed_splits"]["heldout_common"]
    )
    execution["artifact_paths"]["chunk_directory"] = pilot["artifact_paths"][
        "chunk_directory"
    ]
    return execution


def build_pilot_cells(
    pilot: Mapping[str, Any], execution: Mapping[str, Any]
) -> list[runner.CellSpec]:
    cells: list[runner.CellSpec] = []
    count = int(pilot["trajectory_count"])
    for cutoff in pilot["cutoffs"]:
        for scenario in pilot["scenario_names"]:
            horizon = int(
                execution["formal_matrix"]["fault_scenarios"][scenario]["horizon"]
            )
            for backend in ("A", "B"):
                identity = f"pilot|c{cutoff}|fault|{scenario}|{backend}"
                chunk_id = (
                    "".join(character if character.isalnum() else "_" for character in identity)
                    + "__"
                    + sha256(identity.encode("utf-8")).hexdigest()[:16]
                )
                cells.append(
                    runner.CellSpec(
                        chunk_id=chunk_id,
                        layer="fault",
                        cell_base=f"fault|{scenario}",
                        cutoff=int(cutoff),
                        backend=backend,
                        sample_count=count,
                        convergence_role="high_cutoff_state_design_pilot",
                        scenario=scenario,
                        horizon=horizon,
                    )
                )
    if (
        len(cells) != 32
        or len({cell.chunk_id for cell in cells}) != 32
        or sum(cell.expected_rows for cell in cells)
        != 32 * count * 12
    ):
        raise RuntimeError("high-cutoff pilot accounting drift")
    return cells


def _receipt_path(
    root: Path, pilot: Mapping[str, Any], cell: runner.CellSpec
) -> Path:
    return (
        root
        / str(pilot["artifact_paths"]["receipt_directory"])
        / f"{cell.chunk_id}.json"
    )


def _validate_receipt(
    root: Path,
    pilot: Mapping[str, Any],
    cell: runner.CellSpec,
    receipt: Mapping[str, Any],
) -> None:
    unsigned = dict(receipt)
    analysis = unsigned.pop("analysis_sha256", None)
    if (
        receipt.get("task_id") != TASK_ID
        or receipt.get("config_analysis_sha256") != _sha(pilot)
        or receipt.get("cell") != asdict(cell)
        or analysis != _sha(unsigned)
    ):
        raise RuntimeError("pilot receipt identity drift")
    runner._validate_chunk_files(root, receipt, cell)
    for key in ("csv", "npz"):
        binding = receipt.get(key)
        if (
            not isinstance(binding, Mapping)
            or dict(binding)
            != _binding(root / str(binding.get("path")), root)
        ):
            raise RuntimeError(f"pilot {key} binding drift")


def _worker(
    root_text: str,
    pilot: Mapping[str, Any],
    execution: Mapping[str, Any],
    cell_payload: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root_text).resolve()
    cell = runner.CellSpec(**cell_payload)
    receipt_path = _receipt_path(root, pilot, cell)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_receipt(root, pilot, cell, receipt)
        return receipt
    simulator = runner.build_simulators(execution, cell.cutoff)[cell.backend]
    evidence = runner.execute_cell(
        execution, cell, simulator, runner._action_words()
    )
    chunk = runner.write_chunk(root, execution, cell, evidence)
    receipt = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-HIGH-CUTOFF-PILOT-CHUNK-RECEIPT-V1",
        "config_analysis_sha256": _sha(pilot),
        "cell": asdict(cell),
        **chunk,
    }
    receipt["analysis_sha256"] = _sha(receipt)
    _atomic_text(
        receipt_path,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _validate_receipt(root, pilot, cell, receipt)
    return receipt


def _heartbeat(
    root: Path,
    pilot: Mapping[str, Any],
    *,
    completed: int,
    total: int,
    active: bool,
) -> None:
    payload = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-HIGH-CUTOFF-PILOT-HEARTBEAT-V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "completed_cells": completed,
        "expected_cells": total,
        "active": active,
    }
    payload["analysis_sha256"] = _sha(payload)
    _atomic_text(
        root / str(pilot["artifact_paths"]["heartbeat"]),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def run_pilot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    pilot, base = load_pilot_config(root)
    execution = materialize_execution_config(pilot, base)
    cells = build_pilot_cells(pilot, execution)
    manifest_path = root / str(pilot["artifact_paths"]["execution_manifest"])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != STATUS
            or manifest.get("config_analysis_sha256") != _sha(pilot)
            or manifest.get("scientific_verdict") is not None
        ):
            raise RuntimeError("existing pilot manifest is not reusable")
        receipts = manifest.get("chunk_receipts")
        if not isinstance(receipts, list) or len(receipts) != len(cells):
            raise RuntimeError("existing pilot manifest receipt count drift")
        by_id = {receipt["cell"]["chunk_id"]: receipt for receipt in receipts}
        for cell in cells:
            _validate_receipt(root, pilot, cell, by_id[cell.chunk_id])
        return manifest

    _heartbeat(root, pilot, completed=0, total=len(cells), active=True)
    receipts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(pilot["max_workers"])) as executor:
        futures = {
            executor.submit(
                _worker,
                str(root),
                pilot,
                execution,
                asdict(cell),
            ): cell
            for cell in cells
        }
        for future in as_completed(futures):
            receipts.append(future.result())
            _heartbeat(
                root,
                pilot,
                completed=len(receipts),
                total=len(cells),
                active=True,
            )
    by_id = {receipt["cell"]["chunk_id"]: receipt for receipt in receipts}
    ordered = [by_id[cell.chunk_id] for cell in cells]
    for cell, receipt in zip(cells, ordered):
        _validate_receipt(root, pilot, cell, receipt)
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": MANIFEST_SCHEMA,
        "status": STATUS,
        "scientific_verdict": None,
        "qualified_claim": None,
        "config_analysis_sha256": _sha(pilot),
        "observed_cells": len(ordered),
        "observed_rows": sum(cell.expected_rows for cell in cells),
        "exception_rows": 0,
        "chunk_receipts": ordered,
        "claim_state": dict(pilot["claim_boundary"]),
        "bindings": {
            "config": _binding(root / CONFIG_PATH, root),
            "base_config": _binding(
                root / str(pilot["base_config"]["path"]), root
            ),
            "runner_source": _binding(Path(__file__).resolve(), root),
            **{
                name: _binding(root / str(binding["path"]), root)
                for name, binding in pilot["source_bindings"].items()
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    # Exception accounting is recomputed without relying on CSV string shape.
    exception_rows = 0
    for receipt in ordered:
        with (root / str(receipt["csv"]["path"])).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            exception_rows += sum(
                bool(row["exception_type"]) for row in csv.DictReader(stream)
            )
    manifest["exception_rows"] = exception_rows
    manifest["analysis_sha256"] = _sha(manifest)
    _atomic_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _heartbeat(
        root, pilot, completed=len(cells), total=len(cells), active=False
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the state-conditioned high-cutoff design pilot."
    )
    parser.parse_args(argv)
    report = run_pilot(_root())
    print(
        json.dumps(
            {
                "status": report["status"],
                "analysis_sha256": report["analysis_sha256"],
                "observed_cells": report["observed_cells"],
                "observed_rows": report["observed_rows"],
                "exception_rows": report["exception_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "STATUS",
    "build_pilot_cells",
    "load_pilot_config",
    "materialize_execution_config",
    "run_pilot",
]
