"""Minimal outcome-free correction of 16 fault-cutoff gate denominators.

The immutable T06 blueprint accidentally inherited ``stage=round`` from the
generic cutoff-mapping family for fault-trajectory cutoff gates.  T06's own
selected-count record says ``aggregate_fault_clusters=4608`` and T04 freezes
six state-major blocks of 768.  This child changes only count, stage and scope
for the 2 cutoff legs × 4 scenarios × 2 metrics affected by that dispatch bug.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from uuid import uuid4


SCHEMA = "PHASE9-T04-EFFECTIVE-GATE-BLUEPRINT-AMENDMENT-V1"
SOURCE = "docs/t_risk_20260728_06_selected_gate_blueprint.json"
OUTPUT = "docs/t_risk_20260728_04_effective_gate_blueprint.json"
PATTERN = re.compile(
    r"^cutoff/(36-40|40-44)/fault/"
    r"(step|telegraph|burst|compound)/(survival|terminal_density)$"
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("blueprint must be an object")
    return value


def build_amendment(root: Path) -> dict[str, Any]:
    source_path = root / SOURCE
    source = _strict_json(source_path)
    gates = source.get("gates")
    if (
        source.get("schema_version") != "PHASE9-T06-SELECTED-BLUEPRINT-V1"
        or source.get("gate_count") != 3043
        or source.get("stochastic_gate_count") != 3037
        or source.get("selected_count")
        != {
            "aggregate_fault_clusters": 4608,
            "round_clusters": 1536,
            "scale": 2.0,
            "state_clusters": 768,
        }
        or not isinstance(gates, list)
        or len(gates) != 3043
    ):
        raise RuntimeError("immutable T06 blueprint/count parent drifted")
    effective: list[dict[str, Any]] = []
    changes: list[dict[str, object]] = []
    for index, original in enumerate(gates):
        if not isinstance(original, Mapping):
            raise ValueError("gate row is not an object")
        gate = dict(original)
        gate_id = str(gate.get("gate_id"))
        if PATTERN.fullmatch(gate_id):
            before = {
                key: gate.get(key)
                for key in ("cluster_count", "stage", "cluster_scope")
            }
            expected_scope = gate_id.rsplit("/", 1)[0].replace(
                "cutoff/", "trajectory/cutoff/", 1
            )
            if (
                before["cluster_count"] != 1536
                or before["stage"] != "round"
                or before["cluster_scope"] != expected_scope
            ):
                raise RuntimeError(f"fault-cutoff parent shape drift: {gate_id}")
            gate["cluster_count"] = 4608
            gate["stage"] = "trajectory"
            gate["cluster_scope"] = expected_scope + "/all_states"
            changes.append(
                {
                    "gate_id": gate_id,
                    "gate_index": index,
                    "before": before,
                    "after": {
                        key: gate[key]
                        for key in (
                            "cluster_count",
                            "stage",
                            "cluster_scope",
                        )
                    },
                }
            )
        effective.append(gate)
    if len(changes) != 16:
        raise RuntimeError(f"expected exactly 16 dispatch repairs, got {len(changes)}")
    for original, amended in zip(gates, effective, strict=True):
        differences = {
            key
            for key in set(original) | set(amended)
            if original.get(key) != amended.get(key)
        }
        if PATTERN.fullmatch(str(original["gate_id"])):
            if differences != {"cluster_count", "stage", "cluster_scope"}:
                raise RuntimeError("fault-cutoff change whitelist drift")
        elif differences:
            raise RuntimeError("non-target gate changed")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task_id": "T-RISK-20260728-04",
        "parent": _binding(source_path, root),
        "reason": (
            "T06 selected aggregate_fault_clusters=4608; fault-cutoff gates "
            "must consume the frozen six-state 6x768 full denominator"
        ),
        "gate_count": 3043,
        "stochastic_gate_count": 3037,
        "exact_gate_count": 6,
        "changed_gate_count": 16,
        "allowed_changed_fields": [
            "cluster_count",
            "stage",
            "cluster_scope",
        ],
        "changes": changes,
        "gates": effective,
        "formal_outcomes_accessed": False,
        "scientific_margin_changed": False,
        "gate_id_changed": False,
        "gate_deleted": False,
        "postselection_used": False,
        "cross_state_averaging_used": False,
        "qualified_claim": None,
    }
    result["analysis_sha256"] = _sha(result)
    return result


def publish(root: Path) -> dict[str, Any]:
    result = build_amendment(root)
    path = root / OUTPUT
    payload = _canonical(result) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("conflicting immutable effective blueprint")
        return result
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError("effective blueprint publication race")
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    finally:
        if temporary.exists():
            temporary.unlink()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the minimal T04 effective gate blueprint."
    )
    parser.parse_args(list(argv) if argv is not None else None)
    value = publish(_root())
    print(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
