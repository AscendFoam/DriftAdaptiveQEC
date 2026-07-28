"""Independent verifier for the cutoff-32/36 V2 design diagnostic.

This verifier deliberately does not import the diagnostic implementation.  It
rechecks publication bindings, all 30 raw receipt/CSV/NPZ triples, raw
denominators and archive alignment, then independently recomputes every gate's
conservative point, decision, failed-ID ledger and family accounting from the
published source CSV.  It verifies a NO-GO; it cannot release any downstream
experiment or performance claim.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TASK_ID = "T-RISK-20260728-01"
REPORT_PATH = "docs/t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2.json"
SOURCE_PATH = (
    "docs/" "t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2_source_data.csv"
)
COMPLETION_PATH = (
    "docs/" "t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2_completion.json"
)
VERIFICATION_PATH = (
    "docs/" "t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2_verification.json"
)
LAUNCH_META_PATH = (
    "runs/t_risk_20260728_01_cutoff32_36_design_extension_fresh2/"
    "verified_diagnostic_v2_launch_meta.json"
)
RELEASE_CHILD_PATH = (
    "configs/phase9/"
    "t_risk_20260728_01_cutoff32_36_design_diagnostic_v2_released.json"
)
NO_GO_VERDICT = "NO_GO_HIGH_CUTOFF_DESIGN"
VERIFIED_VERDICT = "VERIFIED_NO_GO_HIGH_CUTOFF_DESIGN"
CLAIM_BOUNDARY = {
    "design_extension_only": True,
    "external_sota": None,
    "hardware_measured": None,
    "ler": None,
    "lifetime": None,
    "official_puviani_exact": None,
    "physical_break_even": None,
    "puviani_nmf_surpass": None,
    "twin_qualification": None,
}
SOURCE_FIELDS = (
    "gate_id",
    "family",
    "contrast",
    "scenario",
    "logical_state",
    "stage",
    "metric",
    "cutoff_or_increment",
    "backend_or_pair",
    "estimate",
    "quantization_bound",
    "conservative_point",
    "margin",
    "passed",
    "cluster_count",
    "statistical_role",
    "qualification_effect",
)
EXPECTED_FAMILIES = {
    "fault_absolute_tail": 240,
    "fault_density": 96,
    "fault_scalar": 1080,
    "shared_absolute_tail": 10,
    "shared_density": 7,
    "shared_scalar": 21,
}


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


def _self_hash(document: Mapping[str, Any], label: str) -> str:
    unsigned = dict(document)
    observed = unsigned.pop("analysis_sha256", None)
    expected = _sha(unsigned)
    if observed != expected:
        raise RuntimeError(f"{label} self-hash drift")
    return str(observed)


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _read_bound(root: Path, binding: Mapping[str, Any]) -> tuple[Path, bytes]:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise RuntimeError("verification binding schema drift")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("verification binding escapes repository") from exc
    payload = path.read_bytes()
    if (
        len(payload) != int(binding["bytes"])
        or sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise RuntimeError(f"verification binding drift: {binding['path']}")
    return path, payload


def _load_json(path: Path, label: str) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    _self_hash(document, label)
    return document


def _verify_raw(root: Path, bindings: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if len(bindings) != 90 or len(bindings) % 3:
        raise RuntimeError("raw binding denominator drift")
    global_row_ids: set[str] = set()
    global_density_ids: set[str] = set()
    receipt_count = 0
    row_count = 0
    density_count = 0
    fault_terminal_count = 0
    shared_terminal_count = 0
    for offset in range(0, len(bindings), 3):
        receipt_path, receipt_payload = _read_bound(root, bindings[offset])
        receipt = json.loads(receipt_payload)
        if not isinstance(receipt, dict):
            raise RuntimeError("raw receipt is not a JSON object")
        _self_hash(receipt, f"raw receipt {receipt_path.name}")
        if (
            dict(bindings[offset + 1]) != receipt["csv"]
            or dict(bindings[offset + 2]) != receipt["npz"]
            or receipt["exception_rows"] != 0
            or receipt["observed_rows"] != receipt["expected_rows"]
        ):
            raise RuntimeError("raw receipt completeness/binding drift")
        _, csv_payload = _read_bound(root, bindings[offset + 1])
        _, npz_payload = _read_bound(root, bindings[offset + 2])
        rows = list(
            csv.DictReader(io.StringIO(csv_payload.decode("utf-8"), newline=""))
        )
        if len(rows) != int(receipt["expected_rows"]):
            raise RuntimeError("raw CSV row denominator drift")
        row_ids: list[str] = []
        terminal_ids: list[str] = []
        for index, row in enumerate(rows):
            row_id = row["row_id"]
            if (
                not row_id
                or row_id in global_row_ids
                or int(row["archive_row_index"]) != index
                or int(row["raw_iq_index"]) != index
                or int(row["heldout_iq_index"]) != index
                or row["archive_chunk"] != receipt["chunk_id"]
                or row["exception_type"]
                or row["conservation_pass"] != "True"
            ):
                raise RuntimeError("raw CSV identity/conservation drift")
            global_row_ids.add(row_id)
            row_ids.append(row_id)
            terminal = row["terminal_round"] == "True"
            if terminal:
                if (
                    int(row["density_index"]) != len(terminal_ids)
                    or not row["density_quantization_trace_distance_bound"]
                ):
                    raise RuntimeError("raw terminal density index drift")
                terminal_ids.append(row_id)
                if row["layer"] == "fault":
                    if not row["logical_survival"]:
                        raise RuntimeError("raw fault logical survival missing")
                    fault_terminal_count += 1
                elif row["layer"] == "shared":
                    if row["logical_survival"]:
                        raise RuntimeError("raw shared logical survival must be blank")
                    shared_terminal_count += 1
                else:
                    raise RuntimeError("raw layer drift")
            elif (
                int(row["density_index"]) != -1
                or row["density_quantization_trace_distance_bound"]
            ):
                raise RuntimeError("raw nonterminal density index drift")
        with np.load(io.BytesIO(npz_payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "schema",
                "chunk_id",
                "cutoff",
                "row_ids",
                "density_row_ids",
                "densities",
                "raw_iq",
                "heldout_iq",
            }:
                raise RuntimeError("raw NPZ schema drift")
            archive_row_ids = [str(value) for value in archive["row_ids"]]
            density_ids = [str(value) for value in archive["density_row_ids"]]
            densities = np.asarray(archive["densities"])
            raw_iq = np.asarray(archive["raw_iq"])
            heldout_iq = np.asarray(archive["heldout_iq"])
            cutoff = int(np.asarray(archive["cutoff"])[0])
            schema = str(np.asarray(archive["schema"])[0])
            chunk_id = str(np.asarray(archive["chunk_id"])[0])
        if (
            schema != "PHASE9-FRESH-TWIN-CHUNKED-RAW-ARCHIVE-V1"
            or chunk_id != receipt["chunk_id"]
            or cutoff != int(receipt["cell"]["cutoff"])
            or archive_row_ids != row_ids
            or density_ids != terminal_ids
            or densities.shape != (len(terminal_ids), 3 * cutoff, 3 * cutoff)
            or raw_iq.shape != (len(rows), 8, 2)
            or heldout_iq.shape != (len(rows), 8, 2)
            or not np.all(np.isfinite(densities))
            or not np.all(np.isfinite(raw_iq))
            or not np.all(np.isfinite(heldout_iq))
        ):
            raise RuntimeError("raw NPZ/CSV alignment drift")
        if global_density_ids.intersection(density_ids):
            raise RuntimeError("raw density IDs are not globally unique")
        global_density_ids.update(density_ids)
        receipt_count += 1
        row_count += len(rows)
        density_count += len(density_ids)
    if (
        receipt_count != 30
        or row_count != 21_168
        or density_count != 2_160
        or fault_terminal_count != 1_728
        or shared_terminal_count != 432
    ):
        raise RuntimeError("combined raw denominator drift")
    return {
        "receipt_count": receipt_count,
        "raw_row_count": row_count,
        "density_count": density_count,
        "fault_terminal_count": fault_terminal_count,
        "shared_terminal_count": shared_terminal_count,
    }


def _source_row(raw: Mapping[str, str]) -> dict[str, object]:
    if set(raw) != set(SOURCE_FIELDS):
        raise RuntimeError("diagnostic source schema drift")
    try:
        estimate = float(raw["estimate"])
        quantization = float(raw["quantization_bound"])
        conservative = float(raw["conservative_point"])
        margin = float(raw["margin"])
        clusters = int(raw["cluster_count"])
    except ValueError as exc:
        raise RuntimeError("diagnostic source numeric drift") from exc
    if (
        not all(
            math.isfinite(value)
            for value in (estimate, quantization, conservative, margin)
        )
        or min(estimate, quantization, conservative, margin) < 0.0
        or clusters <= 0
    ):
        raise RuntimeError("diagnostic source numeric domain drift")
    recalculated = estimate + quantization
    if conservative != recalculated:
        raise RuntimeError("diagnostic conservative-point arithmetic drift")
    if raw["passed"] not in {"True", "False"}:
        raise RuntimeError("diagnostic pass token drift")
    passed = raw["passed"] == "True"
    if passed != (recalculated <= margin):
        raise RuntimeError("diagnostic pass/fail recomputation drift")
    return {
        "gate_id": raw["gate_id"],
        "family": raw["family"],
        "contrast": raw["contrast"],
        "scenario": raw["scenario"],
        "logical_state": raw["logical_state"],
        "stage": raw["stage"],
        "metric": raw["metric"],
        "cutoff_or_increment": raw["cutoff_or_increment"],
        "backend_or_pair": raw["backend_or_pair"],
        "estimate": estimate,
        "quantization_bound": quantization,
        "conservative_point": conservative,
        "margin": margin,
        "passed": passed,
        "cluster_count": clusters,
        "statistical_role": raw["statistical_role"],
        "qualification_effect": (
            None if raw["qualification_effect"] == "" else raw["qualification_effect"]
        ),
    }


def _verify_gates(
    report: Mapping[str, Any], source_payload: bytes
) -> dict[str, object]:
    reader = csv.DictReader(io.StringIO(source_payload.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
        raise RuntimeError("diagnostic source header drift")
    rows = [_source_row(raw) for raw in reader]
    gate_ids = [str(row["gate_id"]) for row in rows]
    if (
        len(rows) != 1_454
        or len(set(gate_ids)) != 1_454
        or gate_ids != sorted(gate_ids)
    ):
        raise RuntimeError("diagnostic gate identity/ordering drift")
    if rows != report["gate_rows"]:
        raise RuntimeError("diagnostic source/report gate-row drift")
    family_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row["family"])
        ledger = family_counts.setdefault(
            family, {"total": 0, "passed": 0, "failed": 0}
        )
        ledger["total"] += 1
        ledger["passed" if row["passed"] else "failed"] += 1
    if {
        key: value["total"] for key, value in family_counts.items()
    } != EXPECTED_FAMILIES or family_counts != report["family_counts"]:
        raise RuntimeError("diagnostic family accounting drift")
    failed_ids = [str(row["gate_id"]) for row in rows if row["passed"] is False]
    maximum_ratio = max(
        float(row["conservative_point"]) / float(row["margin"]) for row in rows
    )
    if (
        len(failed_ids) != 61
        or failed_ids != report["failed_gate_ids"]
        or report["failed_gate_count"] != 61
        or report["passed_gate_count"] != 1_393
        or report["gate_count"] != 1_454
        or maximum_ratio != report["maximum_margin_ratio"]
    ):
        raise RuntimeError("diagnostic decision ledger drift")
    return {
        "gate_count": len(rows),
        "passed_gate_count": len(rows) - len(failed_ids),
        "failed_gate_count": len(failed_ids),
        "failed_gate_ids_sha256": _sha(failed_ids),
        "maximum_margin_ratio": maximum_ratio,
        "family_counts": family_counts,
    }


def verify(root: Path | None = None) -> dict[str, Any]:
    repository = _root() if root is None else root.resolve()
    report_path = repository / REPORT_PATH
    source_path = repository / SOURCE_PATH
    completion_path = repository / COMPLETION_PATH
    launch_path = repository / LAUNCH_META_PATH
    child_path = repository / RELEASE_CHILD_PATH
    report = _load_json(report_path, "diagnostic report")
    completion = _load_json(completion_path, "diagnostic completion")
    launch = _load_json(launch_path, "diagnostic launch meta")
    child = _load_json(child_path, "diagnostic V2 released child")
    if (
        report.get("task_id") != TASK_ID
        or report.get("schema_version") != "PHASE9-CUTOFF32-36-DESIGN-DIAGNOSTIC-V2"
        or report.get("scientific_verdict") != NO_GO_VERDICT
        or report.get("authorization_effect") != "POWERED_FORMAL_REMAINS_UNRELEASED"
        or report.get("claim_state") != CLAIM_BOUNDARY
        or report.get("qualified_claim") is not None
        or report.get("equivalence_claim") is not None
        or report.get("formal_outcomes_accessed") is not False
        or completion.get("scientific_verdict") != NO_GO_VERDICT
        or completion.get("claim_state") != CLAIM_BOUNDARY
        or completion.get("qualified_claim") is not None
        or launch.get("mode") != "diagnostic"
        or launch.get("downstream_release") is not False
        or launch.get("qualified_claim") is not None
        or child.get("authorization_state") != "DIAGNOSTIC_LIVE_READER_REPAIR_ONLY"
        or child.get("downstream_release") is not False
        or child.get("gates_or_margins_changed") is not False
        or child.get("design_outcomes_used_to_change_contract") is not False
        or child.get("claim_state") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("diagnostic claim/authorization firewall drift")
    if (
        completion.get("report") != _binding(report_path, repository)
        or completion.get("source_data") != _binding(source_path, repository)
        or report["bindings"]["source_data"] != _binding(source_path, repository)
    ):
        raise RuntimeError("diagnostic publication binding drift")
    source_payload = source_path.read_bytes()
    gate_audit = _verify_gates(report, source_payload)
    raw_bindings = report.get("raw_bindings")
    if (
        not isinstance(raw_bindings, list)
        or report.get("raw_binding_count") != 90
        or report.get("raw_bindings_sha256") != _sha(raw_bindings)
    ):
        raise RuntimeError("raw binding ledger drift")
    raw_audit = _verify_raw(repository, raw_bindings)
    logical = report.get("logical_projection_audit")
    if (
        not isinstance(logical, Mapping)
        or logical.get("fault_terminal_cross_checks") != 1_728
        or logical.get("shared_terminal_derivations") != 432
        or float(logical["maximum_fault_absolute_delta"])
        > float(logical["maximum_fault_allowed_delta"])
    ):
        raise RuntimeError("logical-projection audit drift")
    verification: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": ("PHASE9-CUTOFF32-36-DESIGN-DIAGNOSTIC-V2-VERIFICATION-V1"),
        "status": "COMPLETE",
        "verification_verdict": VERIFIED_VERDICT,
        "scientific_verdict": NO_GO_VERDICT,
        "powered_formal_release": False,
        "gate_audit": gate_audit,
        "raw_audit": raw_audit,
        "logical_projection_audit": dict(logical),
        "bindings": {
            "report": _binding(report_path, repository),
            "source_data": _binding(source_path, repository),
            "completion": _binding(completion_path, repository),
            "launch_meta": _binding(launch_path, repository),
            "released_child": _binding(child_path, repository),
            "verifier_source": _binding(Path(__file__).resolve(), repository),
        },
        "qualified_claim": None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    verification["analysis_sha256"] = _sha(verification)
    return verification


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify the cutoff32/36 V2 NO-GO."
    )
    parser.add_argument("--output", default=VERIFICATION_PATH)
    args = parser.parse_args(argv)
    repository = _root()
    result = verify(repository)
    output = (repository / args.output).resolve()
    output.relative_to(repository)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.write_bytes(payload.encode("utf-8"))
    print(
        json.dumps(
            {
                "verification_verdict": result["verification_verdict"],
                "gate_count": result["gate_audit"]["gate_count"],
                "failed_gate_count": result["gate_audit"]["failed_gate_count"],
                "raw_row_count": result["raw_audit"]["raw_row_count"],
                "analysis_sha256": result["analysis_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
