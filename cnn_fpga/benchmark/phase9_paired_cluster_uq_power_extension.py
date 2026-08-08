"""Fresh-seed power extension of the preserved paired-cluster UQ NO-GO."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import phase9_paired_cluster_uq_calibration as parent


TASK_ID = "T-RISK-20260727-01"
CONFIG_PATH = "configs/phase9/t_risk_20260727_01_uq_power_extension.json"
REPORT_PATH = "docs/t_risk_20260727_01_uq_power_extension.json"
SOURCE_PATH = "docs/t_risk_20260727_01_uq_power_extension_source_data.csv"
SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-POWER-EXTENSION-REPORT-V1"
PASS_VERDICT = "PASS_PAIRED_CLUSTER_UQ_POWER_EXTENSION"
NO_GO_VERDICT = "NO_GO_PAIRED_CLUSTER_UQ_POWER_EXTENSION"


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


def load_extension(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    extension_path = root / CONFIG_PATH
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    if (
        extension.get("task_id") != TASK_ID
        or extension.get("schema_version")
        != "PHASE9-PAIRED-CLUSTER-UQ-POWER-EXTENSION-CONFIG-V1"
        or extension.get("selection_gates_unchanged") is not True
        or extension.get("formal_outcomes_accessed") is not False
        or extension.get("cluster_counts_per_state") != [384, 512]
    ):
        raise ValueError("UQ power-extension identity/outcome firewall invalid")
    for key in ("parent_report", "parent_source_data"):
        expected = extension[key]
        if _binding(root / str(expected["path"]), root) != {
            "path": expected["path"],
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            raise ValueError(f"UQ power-extension {key} binding drift")
    parent_report = json.loads(
        (root / str(extension["parent_report"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        parent_report.get("verdict")
        != extension["parent_report"]["required_verdict"]
        or parent_report.get("analysis_sha256")
        != extension["parent_report"]["required_analysis_sha256"]
        or parent_report.get("selected_formal_clusters_per_state") is not None
        or parent_report.get("selected_calibration_factor")
        != extension["frozen_parent_calibration_factor"]
        or parent_report.get("validation_coverage_summary", {}).get(
            "all_cells_passed"
        )
        is not True
    ):
        raise ValueError("preserved UQ calibration NO-GO semantic binding drift")
    parent_config = json.loads(
        (root / str(extension["parent_config"])).read_text(encoding="utf-8")
    )
    return extension, parent_config, parent_report


def materialize_child_config(
    extension: Mapping[str, Any], parent_config: Mapping[str, Any]
) -> dict[str, Any]:
    child = json.loads(json.dumps(parent_config))
    child["cluster_counts_per_state"] = list(
        extension["cluster_counts_per_state"]
    )
    child["splits"] = json.loads(json.dumps(extension["splits"]))
    child["candidate_calibration_factors"] = [
        float(extension["frozen_parent_calibration_factor"])
    ]
    return child


def _seed_firewall(
    calibration_records: Sequence[Mapping[str, object]],
    validation_records: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    calibration = {int(row["trial_seed"]) for row in calibration_records}
    validation = {int(row["trial_seed"]) for row in validation_records}
    multiplier = {
        int(row["multiplier_seed"])
        for row in (*calibration_records, *validation_records)
    }
    if (
        len(calibration) != len(calibration_records)
        or len(validation) != len(validation_records)
        or len(multiplier)
        != len(calibration_records) + len(validation_records)
        or calibration & validation
        or calibration & multiplier
        or validation & multiplier
    ):
        raise RuntimeError("UQ power-extension materialized seed collision")
    return {
        "calibration_trial_seed_count": len(calibration),
        "validation_trial_seed_count": len(validation),
        "multiplier_seed_count": len(multiplier),
        "collision_count": 0,
    }


def _coverage_passed(
    config: Mapping[str, Any],
    cells: Sequence[Mapping[str, object]],
    *,
    split_name: str,
) -> bool:
    gates = config["selection_gates"]
    if split_name == "calibration":
        rate_key = "calibration_min_cell_coverage"
        lcb_key = "calibration_min_cell_wilson_lcb"
    elif split_name == "validation":
        rate_key = "validation_min_cell_coverage"
        lcb_key = "validation_min_cell_wilson_lcb"
    else:
        raise ValueError("unknown UQ power-extension split")
    return all(
        float(row["coverage_rate"])
        >= float(gates[rate_key])
        and float(row["coverage_wilson_lcb"])
        >= float(gates[lcb_key])
        for row in cells
    )


def build_report(root: Path) -> tuple[dict[str, Any], list[dict[str, object]]]:
    extension, parent_config, parent_report = load_extension(root)
    config = materialize_child_config(extension, parent_config)
    factor = float(extension["frozen_parent_calibration_factor"])
    calibration_records = parent._simulate_split(
        config, split_name="calibration"
    )
    validation_records = parent._simulate_split(
        config, split_name="validation"
    )
    seed_firewall = _seed_firewall(calibration_records, validation_records)
    calibration_cells = parent._coverage_cells(
        calibration_records,
        factor=factor,
        margin=float(config["margin"]),
    )
    validation_cells = parent._coverage_cells(
        validation_records,
        factor=factor,
        margin=float(config["margin"]),
    )
    calibration_coverage_passed = _coverage_passed(
        config, calibration_cells, split_name="calibration"
    )
    validation_coverage_passed = _coverage_passed(
        config, validation_cells, split_name="validation"
    )
    selected_count, count_diagnostics = parent.select_cluster_count(
        config, validation_cells
    )
    verdict = (
        PASS_VERDICT
        if calibration_coverage_passed
        and validation_coverage_passed
        and selected_count is not None
        else NO_GO_VERDICT
    )
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "verdict": verdict,
        "parent_no_go_preserved": True,
        "parent_analysis_sha256": parent_report["analysis_sha256"],
        "frozen_parent_calibration_factor": factor,
        "selected_formal_clusters_per_state": (
            selected_count if verdict == PASS_VERDICT else None
        ),
        "coverage_summary": {
            "calibration_minimum_cell_coverage": min(
                float(row["coverage_rate"]) for row in calibration_cells
            ),
            "calibration_minimum_cell_wilson_lcb": min(
                float(row["coverage_wilson_lcb"]) for row in calibration_cells
            ),
            "validation_minimum_cell_coverage": min(
                float(row["coverage_rate"]) for row in validation_cells
            ),
            "validation_minimum_cell_wilson_lcb": min(
                float(row["coverage_wilson_lcb"]) for row in validation_cells
            ),
            "calibration_all_passed": calibration_coverage_passed,
            "validation_all_passed": validation_coverage_passed,
        },
        "cluster_count_diagnostics": count_diagnostics,
        "seed_firewall": seed_firewall,
        "claim_state": dict(extension["claim_boundary"]),
        "bindings": {
            "extension_config": _binding(root / CONFIG_PATH, root),
            "parent_config": _binding(
                root / str(extension["parent_config"]), root
            ),
            "parent_report": _binding(
                root / str(extension["parent_report"]["path"]), root
            ),
            "parent_source_data": _binding(
                root / str(extension["parent_source_data"]["path"]), root
            ),
            "extension_source": _binding(Path(__file__).resolve(), root),
            "parent_calibration_source": _binding(
                Path(parent.__file__).resolve(), root
            ),
            "paired_cluster_uq_source": _binding(
                root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py", root
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    report["analysis_sha256"] = _sha(report)
    rows = [
        {
            **row,
            "frozen_factor": factor,
            "calibrated_upper_bound": float(row["estimate"])
            + factor * float(row["raw_radius"]),
            "covered": (
                float(row["estimate"])
                + factor * float(row["raw_radius"])
                + 1e-12
                >= float(row["true_distance"])
            ),
            "equivalent": (
                float(row["estimate"])
                + factor * float(row["raw_radius"])
                <= float(config["margin"])
            ),
        }
        for row in (*calibration_records, *validation_records)
    ]
    return report, rows


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


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report, rows = build_report(base)
    _atomic_text(
        base / REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    fields = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        _atomic_text(base / SOURCE_PATH, stream.read())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fresh-seed paired-cluster UQ power extension."
    )
    parser.parse_args(argv)
    report = write_artifacts()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "selected_formal_clusters_per_state": report[
                    "selected_formal_clusters_per_state"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "NO_GO_VERDICT",
    "PASS_VERDICT",
    "REPORT_PATH",
    "SOURCE_PATH",
    "build_report",
    "load_extension",
    "materialize_child_config",
    "write_artifacts",
]
