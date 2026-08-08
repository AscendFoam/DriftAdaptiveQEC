"""Physics-free independent audit of UQ calibration and power extension."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


TASK_ID = "T-RISK-20260727-01"
PARENT_CONFIG = "configs/phase9/t_risk_20260727_01_uq_calibration.json"
PARENT_REPORT = "docs/t_risk_20260727_01_uq_calibration.json"
PARENT_SOURCE = "docs/t_risk_20260727_01_uq_calibration_source_data.csv"
CHILD_CONFIG = "configs/phase9/t_risk_20260727_01_uq_power_extension.json"
CHILD_REPORT = "docs/t_risk_20260727_01_uq_power_extension.json"
CHILD_SOURCE = "docs/t_risk_20260727_01_uq_power_extension_source_data.csv"
AUDIT_REPORT = "docs/t_risk_20260727_01_uq_independent_audit.json"
PASS_VERDICT = "PASS_UQ_CALIBRATION_AND_POWER_EXTENSION_INDEPENDENT_AUDIT"
FAIL_VERDICT = "FAIL_UQ_CALIBRATION_AND_POWER_EXTENSION_AUDIT"


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


def _report_hash_valid(report: Mapping[str, Any]) -> bool:
    unsigned = dict(report)
    analysis = unsigned.pop("analysis_sha256", None)
    return analysis == _sha(unsigned)


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid Boolean field {value!r}")


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            factor_text = raw.get("selected_factor") or raw.get("frozen_factor")
            rows.append(
                {
                    "split": raw["split"],
                    "family": raw["family"],
                    "dimension": int(raw["dimension"]),
                    "cluster_count": int(raw["cluster_count"]),
                    "true_distance": float(raw["true_distance"]),
                    "trial": int(raw["trial"]),
                    "trial_seed": int(raw["trial_seed"]),
                    "multiplier_seed": int(raw["multiplier_seed"]),
                    "estimate": float(raw["estimate"]),
                    "raw_radius": float(raw["raw_radius"]),
                    "factor": float(factor_text),
                    "calibrated_upper_bound": float(
                        raw["calibrated_upper_bound"]
                    ),
                    "covered": _boolean(raw["covered"]),
                    "equivalent": _boolean(raw["equivalent"]),
                    "power_primary": _boolean(raw["power_primary"]),
                }
            )
    return rows


def _wilson_lower(successes: int, total: int) -> float:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def _cell_rows(
    rows: Sequence[Mapping[str, object]], factor: float
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    seen_keys: set[tuple[object, ...]] = set()
    for row in rows:
        unique = (
            row["split"],
            row["family"],
            row["dimension"],
            row["cluster_count"],
            row["true_distance"],
            row["trial"],
        )
        if unique in seen_keys:
            raise ValueError("duplicate UQ trial key")
        seen_keys.add(unique)
        expected_bound = float(row["estimate"]) + factor * float(
            row["raw_radius"]
        )
        if (
            abs(float(row["factor"]) - factor) > 1e-15
            or abs(float(row["calibrated_upper_bound"]) - expected_bound)
            > 2e-14
            or bool(row["covered"])
            != (expected_bound + 1e-12 >= float(row["true_distance"]))
            or bool(row["equivalent"]) != (expected_bound <= 0.1)
        ):
            raise ValueError("UQ source row arithmetic/decision drift")
        key = unique[:-1]
        grouped.setdefault(key, []).append(row)
    cells = []
    for key, values in grouped.items():
        coverage = sum(bool(row["covered"]) for row in values)
        equivalence = sum(bool(row["equivalent"]) for row in values)
        cells.append(
            {
                "split": key[0],
                "family": key[1],
                "dimension": key[2],
                "cluster_count": key[3],
                "true_distance": key[4],
                "trials": len(values),
                "coverage_rate": coverage / len(values),
                "coverage_wilson_lcb": _wilson_lower(coverage, len(values)),
                "equivalence_rate": equivalence / len(values),
                "power_primary": bool(values[0]["power_primary"]),
            }
        )
    return cells


def _seed_firewall(
    rows: Sequence[Mapping[str, object]],
    expected_namespaces: Mapping[str, int],
    multiplier_namespace: int,
) -> bool:
    trial_seeds = [int(row["trial_seed"]) for row in rows]
    multiplier_seeds = [int(row["multiplier_seed"]) for row in rows]
    if (
        len(set(trial_seeds)) != len(trial_seeds)
        or len(set(multiplier_seeds)) != len(multiplier_seeds)
        or set(trial_seeds) & set(multiplier_seeds)
    ):
        return False
    return all(
        int(row["trial_seed"]) >> 64 == int(expected_namespaces[row["split"]])
        and int(row["multiplier_seed"]) >> 64 == multiplier_namespace
        for row in rows
    )


def _coverage_passed(
    cells: Sequence[Mapping[str, object]],
    gates: Mapping[str, Any],
    split: str,
) -> bool:
    if split == "calibration":
        rate = float(gates["calibration_min_cell_coverage"])
        lcb = float(gates["calibration_min_cell_wilson_lcb"])
    elif split == "validation":
        rate = float(gates["validation_min_cell_coverage"])
        lcb = float(gates["validation_min_cell_wilson_lcb"])
    else:
        raise ValueError("unknown audit split")
    selected = [row for row in cells if row["split"] == split]
    return bool(selected) and all(
        float(row["coverage_rate"]) >= rate
        and float(row["coverage_wilson_lcb"]) >= lcb
        for row in selected
    )


def _select_count(
    cells: Sequence[Mapping[str, object]],
    config: Mapping[str, Any],
) -> int | None:
    gates = config["selection_gates"]
    for count in config["cluster_counts_per_state"]:
        primary = [
            row
            for row in cells
            if row["split"] == "validation"
            and row["power_primary"]
            and int(row["cluster_count"]) == int(count)
        ]
        rates = {
            float(effect): min(
                float(row["equivalence_rate"])
                for row in primary
                if abs(float(row["true_distance"]) - float(effect)) < 1e-12
            )
            for effect in config["true_trace_distances"]
        }
        if (
            rates[0.0] >= float(gates["null_min_equivalence_rate"])
            and rates[0.05]
            >= float(gates["local_005_min_equivalence_rate"])
            and rates[0.1] <= float(gates["boundary_max_equivalence_rate"])
            and rates[0.12] <= float(gates["outside_max_equivalence_rate"])
        ):
            return int(count)
    return None


def _claims_scoped_null(
    values: Mapping[str, object], marker: str
) -> bool:
    return (
        values.get(marker) is True
        and all(value is None for key, value in values.items() if key != marker)
    )


def audit_bundle(root: Path) -> dict[str, Any]:
    parent_config = json.loads((root / PARENT_CONFIG).read_text(encoding="utf-8"))
    parent_report = json.loads((root / PARENT_REPORT).read_text(encoding="utf-8"))
    child_config_raw = json.loads((root / CHILD_CONFIG).read_text(encoding="utf-8"))
    child_report = json.loads((root / CHILD_REPORT).read_text(encoding="utf-8"))
    parent_rows = _load_rows(root / PARENT_SOURCE)
    child_rows = _load_rows(root / CHILD_SOURCE)
    child_config = json.loads(json.dumps(parent_config))
    child_config["cluster_counts_per_state"] = child_config_raw[
        "cluster_counts_per_state"
    ]
    child_config["splits"] = child_config_raw["splits"]
    factor = float(parent_report["selected_calibration_factor"])
    parent_cells = _cell_rows(parent_rows, factor)
    child_cells = _cell_rows(child_rows, factor)

    gates: list[dict[str, object]] = []

    def gate(gate_id: str, passed: bool, detail: object) -> None:
        gates.append({"gate_id": gate_id, "passed": bool(passed), "detail": detail})

    gate("A01_parent_report_hash", _report_hash_valid(parent_report), None)
    gate("A02_child_report_hash", _report_hash_valid(child_report), None)
    gate(
        "A03_parent_denominator",
        len(parent_rows) == 19200,
        len(parent_rows),
    )
    gate(
        "A04_child_denominator",
        len(child_rows) == 12800,
        len(child_rows),
    )
    gate(
        "A05_parent_seed_firewall",
        _seed_firewall(
            parent_rows,
            {
                "calibration": int(
                    parent_config["splits"]["calibration"]["seed_base"]
                ),
                "validation": int(
                    parent_config["splits"]["validation"]["seed_base"]
                ),
            },
            int(parent_config["splits"]["multiplier_seed_base"]),
        ),
        None,
    )
    gate(
        "A06_child_seed_firewall",
        _seed_firewall(
            child_rows,
            {
                "calibration": int(
                    child_config["splits"]["calibration"]["seed_base"]
                ),
                "validation": int(
                    child_config["splits"]["validation"]["seed_base"]
                ),
            },
            int(child_config["splits"]["multiplier_seed_base"]),
        ),
        None,
    )
    gate(
        "A07_parent_coverage",
        _coverage_passed(
            parent_cells, parent_config["selection_gates"], "calibration"
        )
        and _coverage_passed(
            parent_cells, parent_config["selection_gates"], "validation"
        ),
        None,
    )
    gate(
        "A08_child_coverage",
        _coverage_passed(
            child_cells, child_config["selection_gates"], "calibration"
        )
        and _coverage_passed(
            child_cells, child_config["selection_gates"], "validation"
        ),
        None,
    )
    parent_count = _select_count(parent_cells, parent_config)
    child_count = _select_count(child_cells, child_config)
    gate(
        "A09_parent_no_go_recomputed",
        parent_count is None
        and parent_report["verdict"]
        == "NO_GO_PAIRED_CLUSTER_UQ_CALIBRATION"
        and parent_report["selected_formal_clusters_per_state"] is None,
        parent_count,
    )
    gate(
        "A10_child_count_recomputed",
        child_count is not None
        and child_report["verdict"]
        == "PASS_PAIRED_CLUSTER_UQ_POWER_EXTENSION"
        and child_report["selected_formal_clusters_per_state"] == child_count,
        child_count,
    )
    gate(
        "A11_parent_no_go_preserved",
        child_report.get("parent_no_go_preserved") is True
        and child_report.get("parent_analysis_sha256")
        == parent_report.get("analysis_sha256"),
        child_report.get("parent_analysis_sha256"),
    )
    gate(
        "A12_claim_boundaries",
        _claims_scoped_null(parent_report["claim_state"], "calibration_only")
        and _claims_scoped_null(
            child_report["claim_state"], "power_extension_only"
        ),
        None,
    )
    for prefix, report in (("parent", parent_report), ("child", child_report)):
        for name, binding in report["bindings"].items():
            gate(
                f"A13_{prefix}_binding_{name}",
                _binding(root / str(binding["path"]), root) == dict(binding),
                binding["path"],
            )
    verdict = PASS_VERDICT if all(row["passed"] for row in gates) else FAIL_VERDICT
    audit: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-UQ-INDEPENDENT-AUDIT-V1",
        "verdict": verdict,
        "parent_no_go_preserved": True,
        "recomputed_parent_selected_count": parent_count,
        "recomputed_child_selected_count": child_count,
        "gate_summary": {
            "passed": sum(bool(row["passed"]) for row in gates),
            "failed": sum(not bool(row["passed"]) for row in gates),
            "total": len(gates),
        },
        "gates": gates,
        "claim_state": {
            "twin_qualification": None,
            "ler": None,
            "lifetime": None,
            "physical_break_even": None,
            "official_puviani_exact": None,
            "puviani_nmf_surpass": None,
            "external_sota": None,
            "hardware_measured": None,
        },
        "bindings": {
            "parent_report": _binding(root / PARENT_REPORT, root),
            "parent_source": _binding(root / PARENT_SOURCE, root),
            "child_report": _binding(root / CHILD_REPORT, root),
            "child_source": _binding(root / CHILD_SOURCE, root),
            "audit_source": _binding(Path(__file__).resolve(), root),
        },
    }
    audit["analysis_sha256"] = _sha(audit)
    return audit


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


def write_audit(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report = audit_bundle(base)
    _atomic_text(
        base / AUDIT_REPORT,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently audit Phase-9 UQ calibration evidence."
    )
    parser.parse_args(argv)
    report = write_audit()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "gate_summary": report["gate_summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_REPORT",
    "FAIL_VERDICT",
    "PASS_VERDICT",
    "audit_bundle",
    "write_audit",
]
