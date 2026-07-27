"""Independent post-outcome audit for the fresh Phase-9 twin qualification.

The auditor imports neither physics, the formal runner, nor the formal
verifier.  It independently checks byte bindings, the append-only attempt
chain, every archived chunk hash, the complete row denominator, all 1,589
TOST gates, the scientific NO-GO release policy, typed-null claims, historical
NO-GO preservation, and Git-LFS tracking of the two large evidence objects.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


TASK_ID = "T-RISK-20260726-01"
SCHEMA_VERSION = "PHASE9-FRESH-TWIN-POST-OUTCOME-AUDIT-V1"
VERDICT = "AUDIT_CONFIRMS_FRESH_SCIENTIFIC_NO_GO"
FORMAL_PASS = "PASS_FRESH_TWIN_QUALIFIED"
FORMAL_NO_GO = "NO_GO_FRESH_TWIN_QUALIFICATION"
FORMAL_INCOMPLETE = "INCOMPLETE_FAIL_CLOSED"
QUALIFIED_CLAIM = "dual_backend_agreement_for_fresh_repaired_synthetic_task"
Z_TOST = 1.6448536269514722

CLAIM_FIELDS = (
    "frontend_performance",
    "synthetic_iq_qualification",
    "recorded_iq_qualification",
    "live_raw_iq_qualification",
    "board_measured_latency",
    "board_resources",
    "board_power",
    "external_same_task_speed",
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "rank",
)
HISTORICAL_CLAIM_FIELDS = (
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "hardware_measured",
    "rank",
)
DOWNSTREAM_TASKS = (
    "T9.2.5",
    "T9.2.7",
    "T9.3.1",
    "T9.3.4",
    "T9.6.2",
    "T9.6.5",
)
EXPECTED_FAMILY_TOTALS = {
    "cutoff_mapping": 88,
    "fault_trajectory_tail": 108,
    "iq_conditional_distribution": 484,
    "likelihood_score_posterior": 363,
    "logical_ptm_survival": 147,
    "physical_state_channel": 363,
    "reset_leakage": 36,
}
EXPECTED_FAILED_FAMILIES = {
    "cutoff_mapping": 8,
    "fault_trajectory_tail": 16,
    "physical_state_channel": 3,
}
EXPECTED_ARTIFACTS = {
    "config": "configs/phase9/t_risk_20260726_01_fresh_twin_qualification.json",
    "design_config": "configs/phase9/t_risk_20260726_01_design_power.json",
    "preformal_audit": "docs/t_risk_20260726_01_fresh_preformal_audit.json",
    "preformal_seal": "docs/t_risk_20260726_01_fresh_preformal_seal.json",
    "attempt_ledger": "docs/t_risk_20260726_01_fresh_attempt_ledger.jsonl",
    "cell_ledger": "docs/t_risk_20260726_01_fresh_cell_ledger.csv",
    "raw_archive": "docs/t_risk_20260726_01_fresh_raw_archive.zip",
    "execution_manifest": "docs/t_risk_20260726_01_fresh_execution_manifest.json",
    "qualification": "docs/t_risk_20260726_01_fresh_qualification.json",
    "qualification_source": (
        "docs/t_risk_20260726_01_fresh_qualification_source_data.csv"
    ),
    "gate_ledger": "docs/t_risk_20260726_01_fresh_gate_ledger.csv",
    "verification": "docs/t_risk_20260726_01_fresh_verification.json",
    "release": "docs/t_risk_20260726_01_fresh_release.json",
    "release_pin": "configs/phase9/t_risk_20260726_01_fresh_release_pin.json",
    "historical_lineage": (
        "docs/t_risk_20260726_01_historical_no_go_receipt.json"
    ),
    "historical_report": "docs/t9_2_4_dual_backend_qualification.json",
    "gitattributes": ".gitattributes",
    "auditor": (
        "cnn_fpga/benchmark/phase9_fresh_twin_post_outcome_audit.py"
    ),
    "auditor_tests": (
        "tests/test_phase9_fresh_twin_post_outcome_audit.py"
    ),
}
OUTPUT_PATH = "docs/t_risk_20260726_01_fresh_post_outcome_audit.json"
SOURCE_PATH = (
    "docs/t_risk_20260726_01_fresh_post_outcome_audit_source_data.csv"
)


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


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha(value: object) -> str:
    return _sha_bytes(_canonical(value))


def _sha_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _binding(root: Path, relative: str) -> dict[str, object]:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes repository: {relative}")
    size, digest = _sha_file(path)
    return {"path": relative, "bytes": size, "sha256": digest}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _self_hash_valid(
    document: Mapping[str, Any],
    field: str = "analysis_sha256",
) -> bool:
    claimed = document.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        return False
    unsigned = dict(document)
    unsigned.pop(field, None)
    return _sha(unsigned) == claimed


def _binding_matches(
    root: Path,
    binding: object,
    *,
    expected_path: str | None = None,
) -> bool:
    if not isinstance(binding, dict):
        return False
    relative = binding.get("path")
    if not isinstance(relative, str):
        return False
    if expected_path is not None and relative != expected_path:
        return False
    try:
        return _binding(root, relative) == binding
    except (OSError, ValueError):
        return False


def _path_sha_binding_matches(
    root: Path,
    binding: object,
    *,
    expected_path: str,
) -> bool:
    """Validate a legacy immutable binding that carries metadata but no size."""
    if not isinstance(binding, dict):
        return False
    if binding.get("path") != expected_path:
        return False
    claimed = binding.get("sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        return False
    try:
        return _sha_file(root / expected_path)[1] == claimed
    except OSError:
        return False


class _HashingRaw(io.RawIOBase):
    def __init__(self, raw: io.BufferedReader) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.size = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        count = self.raw.readinto(buffer)
        if count:
            payload = memoryview(buffer)[:count]
            self.digest.update(payload)
            self.size += count
        return count


def _scan_cell_ledger(path: Path) -> dict[str, Any]:
    raw_file = path.open("rb")
    hashing = _HashingRaw(raw_file)
    buffered = io.BufferedReader(hashing, buffer_size=8 * 1024 * 1024)
    text = io.TextIOWrapper(buffered, encoding="utf-8", newline="")
    row_ids: set[str] = set()
    chunk_counts: dict[str, int] = {}
    exception_rows = 0
    conservation_rows = 0
    nonfinite_tokens = 0
    row_count = 0
    required = {
        "row_id",
        "archive_chunk",
        "backend",
        "cutoff",
        "seed_position",
        "conservation_pass",
        "exception_type",
    }
    try:
        reader = csv.DictReader(text)
        header = list(reader.fieldnames or [])
        if not required.issubset(header):
            raise ValueError("cell ledger required header missing")
        for row in reader:
            row_count += 1
            row_id = row["row_id"]
            row_ids.add(row_id)
            chunk = row["archive_chunk"]
            chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
            exception_rows += bool(row["exception_type"])
            conservation_rows += row["conservation_pass"] == "True"
            # These fields are mandatory finite scalars for every retained row.
            for field in ("cutoff", "seed", "seed_position", "round_index"):
                token = row[field].strip().lower()
                if token in {
                    "nan",
                    "+nan",
                    "-nan",
                    "inf",
                    "+inf",
                    "-inf",
                    "infinity",
                }:
                    nonfinite_tokens += 1
    finally:
        text.close()
    return {
        "bytes": hashing.size,
        "sha256": hashing.digest.hexdigest(),
        "header": header,
        "header_sha256": _sha(header),
        "row_count": row_count,
        "unique_row_ids": len(row_ids),
        "chunk_count": len(chunk_counts),
        "chunk_counts": chunk_counts,
        "exception_rows": exception_rows,
        "conservation_rows": conservation_rows,
        "mandatory_nonfinite_tokens": nonfinite_tokens,
    }


def _scan_attempt_ledger(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    full_digest = hashlib.sha256()
    prefix_digest = hashlib.sha256()
    prefix_bytes = 0
    previous = "0" * 64
    chain_valid = True
    indexes_valid = True
    chunk_receipts: dict[str, dict[str, Any]] = {}
    run_errors = 0
    final_event: dict[str, Any] | None = None
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream):
            full_digest.update(raw_line)
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"attempt ledger line {line_number + 1} invalid"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError("attempt event must be an object")
            if event.get("event_index") != line_number:
                indexes_valid = False
            if event.get("previous_event_sha256") != previous:
                chain_valid = False
            unsigned = dict(event)
            claimed = unsigned.pop("event_sha256", None)
            actual = _sha(unsigned)
            if claimed != actual:
                chain_valid = False
            previous = str(claimed)
            kind = event.get("event_kind")
            if kind == "CHUNK_COMMITTED":
                chunk = event.get("chunk")
                if not isinstance(chunk, dict):
                    raise ValueError("chunk event payload missing")
                chunk_id = str(chunk.get("chunk_id"))
                if chunk_id in chunk_receipts:
                    chain_valid = False
                chunk_receipts[chunk_id] = chunk
            elif kind == "RUN_ERROR":
                run_errors += 1
            elif kind == "FINALIZED":
                final_event = event
            if kind != "FINALIZED":
                prefix_digest.update(raw_line)
                prefix_bytes += len(raw_line)
            events.append(event)
    kinds = [event.get("event_kind") for event in events]
    chunk_rows = sum(
        int(chunk.get("observed_rows", -1))
        for chunk in chunk_receipts.values()
    )
    chunk_exceptions = sum(
        int(chunk.get("exception_rows", -1))
        for chunk in chunk_receipts.values()
    )
    chunks_complete = all(
        chunk.get("expected_rows") == chunk.get("observed_rows")
        for chunk in chunk_receipts.values()
    )
    return {
        "bytes": path.stat().st_size,
        "sha256": full_digest.hexdigest(),
        "event_count": len(events),
        "chain_valid": chain_valid,
        "indexes_valid": indexes_valid,
        "first_kind": kinds[0] if kinds else None,
        "final_kind": kinds[-1] if kinds else None,
        "run_error_count": run_errors,
        "chunk_count": len(chunk_receipts),
        "chunk_rows": chunk_rows,
        "chunk_exceptions": chunk_exceptions,
        "chunks_complete": chunks_complete,
        "chunk_receipts": chunk_receipts,
        "prefix_bytes": prefix_bytes,
        "prefix_sha256": prefix_digest.hexdigest(),
        "prefix_last_event_index": len(events) - 2,
        "prefix_last_event_sha256": (
            events[-2].get("event_sha256") if len(events) >= 2 else None
        ),
        "final_event": final_event,
    }


def _scan_archive(
    path: Path,
    attempt_chunks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicate_names = len(names) - len(set(names))
        try:
            manifest = json.loads(
                archive.read("archive_manifest.json").decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("archive manifest missing/corrupt") from exc
        if not isinstance(manifest, dict):
            raise ValueError("archive manifest must be an object")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("archive entries missing")
        member_hash_failures: list[str] = []
        member_size_failures: list[str] = []
        attempt_binding_failures: list[str] = []
        bad_crc_member: str | None = None
        entry_ids: list[str] = []
        entry_rows = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("archive entry invalid")
            chunk_id = str(entry.get("chunk_id"))
            member = str(entry.get("member"))
            source = entry.get("source")
            if not isinstance(source, dict):
                raise ValueError("archive entry source missing")
            entry_ids.append(chunk_id)
            entry_rows += int(entry.get("rows", -1))
            digest = hashlib.sha256()
            size = 0
            try:
                with archive.open(member) as stream:
                    while True:
                        block = stream.read(8 * 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        size += len(block)
            except KeyError:
                member_hash_failures.append(chunk_id)
                continue
            except zipfile.BadZipFile:
                bad_crc_member = member
                member_hash_failures.append(chunk_id)
                continue
            if digest.hexdigest() != source.get("sha256"):
                member_hash_failures.append(chunk_id)
            if size != source.get("bytes"):
                member_size_failures.append(chunk_id)
            attempt = attempt_chunks.get(chunk_id)
            if (
                not isinstance(attempt, dict)
                or attempt.get("npz") != source
                or attempt.get("observed_rows") != entry.get("rows")
            ):
                attempt_binding_failures.append(chunk_id)
        mapping = manifest.get("mapping_source")
        mapping_member = manifest.get("mapping_member")
        mapping_hash_valid = False
        if isinstance(mapping, dict) and isinstance(mapping_member, str):
            digest = hashlib.sha256()
            size = 0
            try:
                with archive.open(mapping_member) as stream:
                    while True:
                        block = stream.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        size += len(block)
                mapping_hash_valid = (
                    digest.hexdigest() == mapping.get("sha256")
                    and size == mapping.get("bytes")
                )
            except zipfile.BadZipFile:
                bad_crc_member = mapping_member
        expected_names = {
            "archive_manifest.json",
            str(mapping_member),
            *(str(entry.get("member")) for entry in entries),
        }
        # Reading each member to EOF above makes ZipExtFile validate its CRC.
        # Calling ``testzip`` here would redundantly decompress all 824 MB.
    return {
        "manifest": manifest,
        "manifest_self_hash_valid": _self_hash_valid(manifest),
        "zip_member_count": len(names),
        "duplicate_member_names": duplicate_names,
        "member_name_set_exact": set(names) == expected_names,
        "bad_crc_member": bad_crc_member,
        "entry_count": len(entries),
        "unique_entry_ids": len(set(entry_ids)),
        "entry_ids": entry_ids,
        "entry_rows": entry_rows,
        "member_hash_failures": member_hash_failures,
        "member_size_failures": member_size_failures,
        "attempt_binding_failures": attempt_binding_failures,
        "mapping_hash_valid": mapping_hash_valid,
    }


def _scan_gate_ledger(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    identifiers: list[str] = []
    family_totals: dict[str, int] = {}
    family_failures: dict[str, int] = {}
    arithmetic_failures: list[str] = []
    declared_pass_failures: list[str] = []
    nonfinite_rows: list[str] = []
    no_postselection = True
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        header = list(reader.fieldnames or [])
        for raw in reader:
            gate_id = raw["gate_id"]
            identifiers.append(gate_id)
            family = raw["family"]
            family_totals[family] = family_totals.get(family, 0) + 1
            try:
                estimate = float(raw["estimate"])
                standard_error = float(raw["standard_error"])
                bound = float(raw["bound"])
                margin = float(raw["margin"])
            except ValueError:
                nonfinite_rows.append(gate_id)
                continue
            if not all(
                math.isfinite(value)
                for value in (estimate, standard_error, bound, margin)
            ):
                nonfinite_rows.append(gate_id)
                continue
            direction = raw["direction"]
            expected_bound = (
                estimate - Z_TOST * standard_error
                if direction == "lower"
                else abs(estimate) + Z_TOST * standard_error
            )
            if not math.isclose(
                bound, expected_bound, rel_tol=0.0, abs_tol=2e-12
            ):
                arithmetic_failures.append(gate_id)
            expected_pass = (
                bound >= margin if direction == "lower" else bound <= margin
            )
            declared_pass = raw["passed"] == "True"
            if expected_pass != declared_pass:
                declared_pass_failures.append(gate_id)
            if not declared_pass:
                family_failures[family] = family_failures.get(family, 0) + 1
            denominator = raw["denominator"].lower()
            if (
                not denominator
                or "postselect" in denominator
                or "accepted-only" in denominator
            ):
                no_postselection = False
            rows.append(
                {
                    "gate_id": gate_id,
                    "family": family,
                    "stage": raw["stage"],
                    "metric": raw["metric"],
                    "direction": direction,
                    "margin": margin,
                    "passed": declared_pass,
                }
            )
    return {
        "header": header,
        "rows": rows,
        "row_count": len(rows),
        "unique_gate_ids": len(set(identifiers)),
        "family_totals": dict(sorted(family_totals.items())),
        "family_failures": dict(sorted(family_failures.items())),
        "failed_ids": [
            row["gate_id"] for row in rows if row["passed"] is False
        ],
        "passed_count": sum(row["passed"] is True for row in rows),
        "failed_count": sum(row["passed"] is False for row in rows),
        "arithmetic_failures": arithmetic_failures,
        "declared_pass_failures": declared_pass_failures,
        "nonfinite_rows": nonfinite_rows,
        "no_postselection_denominators": no_postselection,
    }


def _blueprint_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    wrapper = config.get("gate_blueprint")
    if not isinstance(wrapper, dict):
        return []
    rows = wrapper.get("rows")
    return rows if isinstance(rows, list) else []


def _gate_blueprint_matches(
    gates: Sequence[Mapping[str, Any]],
    blueprint: Sequence[Mapping[str, Any]],
) -> bool:
    if len(gates) != len(blueprint):
        return False
    fields = ("gate_id", "family", "stage", "metric", "direction")
    for gate, spec in zip(gates, blueprint):
        if any(gate.get(field) != spec.get(field) for field in fields):
            return False
        try:
            if not math.isclose(
                float(gate["margin"]),
                float(spec["margin"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _claims_literal_null(document: Mapping[str, Any]) -> bool:
    state = document.get("claim_state")
    return (
        isinstance(state, dict)
        and set(state) == set(CLAIM_FIELDS)
        and all(value is None for value in state.values())
    )


def _historical_claims_literal_null(
    document: Mapping[str, Any],
) -> bool:
    state = document.get("claim_state")
    return (
        isinstance(state, dict)
        and set(state) == set(HISTORICAL_CLAIM_FIELDS)
        and all(value is None for value in state.values())
    )


def _release_all_false(release: Mapping[str, Any]) -> bool:
    downstream = release.get("downstream_release")
    return (
        isinstance(downstream, dict)
        and tuple(downstream) == DOWNSTREAM_TASKS
        and all(
            isinstance(row, dict)
            and row.get("released") is False
            and row.get("reason") == "fresh twin qualification did not pass"
            for row in downstream.values()
        )
    )


def _lfs_tracking(root: Path) -> dict[str, Any]:
    text = (root / ".gitattributes").read_text(encoding="utf-8")
    paths = (
        EXPECTED_ARTIFACTS["cell_ledger"],
        EXPECTED_ARTIFACTS["raw_archive"],
    )
    rows = {
        path: (
            f"{path} filter=lfs diff=lfs merge=lfs -text" in text.splitlines()
        )
        for path in paths
    }
    return {
        "paths": rows,
        "all_tracked": all(rows.values()),
    }


def _branch_fixture(branch: str) -> dict[str, Any]:
    claims = {field: None for field in CLAIM_FIELDS}
    if branch == "pass":
        return {
            "verdict": FORMAL_PASS,
            "complete": True,
            "all_gates_passed": True,
            "qualified_claim": QUALIFIED_CLAIM,
            "claim_state": claims,
            "released": {task: True for task in DOWNSTREAM_TASKS},
            "historical_no_go_preserved": True,
        }
    if branch == "no_go":
        return {
            "verdict": FORMAL_NO_GO,
            "complete": True,
            "all_gates_passed": False,
            "qualified_claim": None,
            "claim_state": claims,
            "released": {task: False for task in DOWNSTREAM_TASKS},
            "historical_no_go_preserved": True,
        }
    if branch == "incomplete":
        return {
            "verdict": FORMAL_INCOMPLETE,
            "complete": False,
            "all_gates_passed": False,
            "qualified_claim": None,
            "claim_state": claims,
            "released": {task: False for task in DOWNSTREAM_TASKS},
            "historical_no_go_preserved": True,
        }
    raise ValueError(f"unknown branch: {branch}")


def audit_branch_fixture(fixture: Mapping[str, Any]) -> dict[str, bool]:
    verdict = fixture.get("verdict")
    claims = fixture.get("claim_state")
    released = fixture.get("released")
    exact_claims = (
        isinstance(claims, dict)
        and set(claims) == set(CLAIM_FIELDS)
        and all(value is None for value in claims.values())
    )
    release_schema = (
        isinstance(released, dict)
        and tuple(released) == DOWNSTREAM_TASKS
        and all(isinstance(value, bool) for value in released.values())
    )
    pass_branch = verdict != FORMAL_PASS or (
        fixture.get("complete") is True
        and fixture.get("all_gates_passed") is True
        and fixture.get("qualified_claim") == QUALIFIED_CLAIM
        and release_schema
        and all(released.values())
    )
    no_go_branch = verdict != FORMAL_NO_GO or (
        fixture.get("complete") is True
        and fixture.get("all_gates_passed") is False
        and fixture.get("qualified_claim") is None
        and release_schema
        and not any(released.values())
    )
    incomplete_branch = verdict != FORMAL_INCOMPLETE or (
        fixture.get("complete") is False
        and fixture.get("all_gates_passed") is False
        and fixture.get("qualified_claim") is None
        and release_schema
        and not any(released.values())
    )
    return {
        "F01_known_branch": verdict
        in {FORMAL_PASS, FORMAL_NO_GO, FORMAL_INCOMPLETE},
        "F02_fifteen_typed_null": exact_claims,
        "F03_pass_release": pass_branch,
        "F04_no_go_block": no_go_branch,
        "F05_incomplete_block": incomplete_branch,
        "F06_historical_no_go_preserved": fixture.get(
            "historical_no_go_preserved"
        )
        is True,
    }


def build_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    documents = {
        name: _load_json(root / path)
        for name, path in EXPECTED_ARTIFACTS.items()
        if path.endswith(".json")
    }
    bindings = {
        name: _binding(root, path)
        for name, path in EXPECTED_ARTIFACTS.items()
    }
    manifest = documents["execution_manifest"]
    qualification = documents["qualification"]
    verification = documents["verification"]
    release = documents["release"]
    release_pin = documents["release_pin"]
    config = documents["config"]
    lineage = documents["historical_lineage"]
    historical = documents["historical_report"]

    attempt = _scan_attempt_ledger(
        root / EXPECTED_ARTIFACTS["attempt_ledger"]
    )
    ledger = _scan_cell_ledger(
        root / EXPECTED_ARTIFACTS["cell_ledger"]
    )
    archive = _scan_archive(
        root / EXPECTED_ARTIFACTS["raw_archive"],
        attempt["chunk_receipts"],
    )
    gate = _scan_gate_ledger(
        root / EXPECTED_ARTIFACTS["gate_ledger"]
    )
    blueprint = _blueprint_rows(config)
    final_event = attempt["final_event"] or {}
    attempt_prefix = manifest.get("attempt_ledger_prefix", {})
    archive_manifest = archive["manifest"]
    historical_binding = lineage.get("historical_bindings", {}).get(
        "formal_report"
    )
    report_bindings = qualification.get("bindings", {})
    verification_bindings = verification.get("bindings", {})

    chunk_counts_match = all(
        ledger["chunk_counts"].get(chunk_id)
        == int(receipt.get("observed_rows", -1))
        for chunk_id, receipt in attempt["chunk_receipts"].items()
    ) and set(ledger["chunk_counts"]) == set(attempt["chunk_receipts"])

    final_bindings_match = all(
        _binding_matches(
            root,
            final_event.get(key),
            expected_path=EXPECTED_ARTIFACTS[
                {
                    "cell_ledger": "cell_ledger",
                    "raw_archive": "raw_archive",
                    "execution_manifest": "execution_manifest",
                }[key]
            ],
        )
        for key in ("cell_ledger", "raw_archive", "execution_manifest")
    )
    report_bindings_match = all(
        _binding_matches(root, binding)
        for binding in report_bindings.values()
    )
    verification_bindings_match = all(
        _binding_matches(root, binding)
        for binding in verification_bindings.values()
    )

    family_summary = qualification.get("family_summary", {})
    report_family_totals = {
        family: row.get("total")
        for family, row in family_summary.items()
        if isinstance(row, dict)
    }
    report_family_failures = {
        family: int(row.get("total", 0)) - int(row.get("passed", 0))
        for family, row in family_summary.items()
        if isinstance(row, dict)
        and int(row.get("total", 0)) - int(row.get("passed", 0)) > 0
    }
    semantic = qualification.get("independent_semantic_receipt_check", {})
    lfs = _lfs_tracking(root)

    return {
        "task_ids_exact": all(
            document.get("task_id") == TASK_ID
            for name, document in documents.items()
            if name not in {"historical_report"}
        ),
        "manifest_self_hash": _self_hash_valid(
            manifest, field="execution_sha256"
        ),
        "qualification_self_hash": _self_hash_valid(qualification),
        "verification_self_hash": _self_hash_valid(verification),
        "release_self_hash": _self_hash_valid(release),
        "release_pin_self_hash": _self_hash_valid(release_pin),
        "lineage_self_hash": _self_hash_valid(lineage),
        "historical_self_hash": _self_hash_valid(historical),
        "report_bindings_match": report_bindings_match,
        "verification_bindings_match": verification_bindings_match,
        "manifest_config_binding": (
            manifest.get("config") == bindings["config"]
        ),
        "manifest_seal_binding": (
            manifest.get("preformal_seal") == bindings["preformal_seal"]
        ),
        "manifest_ledger_binding": (
            manifest.get("cell_ledger") == bindings["cell_ledger"]
        ),
        "manifest_archive_binding": (
            manifest.get("raw_archive") == bindings["raw_archive"]
        ),
        "row_count": ledger["row_count"],
        "row_unique": ledger["unique_row_ids"],
        "row_bytes": ledger["bytes"],
        "row_sha256": ledger["sha256"],
        "ledger_header_sha256": ledger["header_sha256"],
        "ledger_chunk_count": ledger["chunk_count"],
        "ledger_exception_rows": ledger["exception_rows"],
        "ledger_conservation_rows": ledger["conservation_rows"],
        "ledger_mandatory_nonfinite": ledger[
            "mandatory_nonfinite_tokens"
        ],
        "chunk_counts_match": chunk_counts_match,
        "attempt_event_count": attempt["event_count"],
        "attempt_chain_valid": attempt["chain_valid"],
        "attempt_indexes_valid": attempt["indexes_valid"],
        "attempt_first_kind": attempt["first_kind"],
        "attempt_final_kind": attempt["final_kind"],
        "attempt_run_errors": attempt["run_error_count"],
        "attempt_chunk_count": attempt["chunk_count"],
        "attempt_chunk_rows": attempt["chunk_rows"],
        "attempt_chunk_exceptions": attempt["chunk_exceptions"],
        "attempt_chunks_complete": attempt["chunks_complete"],
        "attempt_prefix_bytes": attempt["prefix_bytes"],
        "attempt_prefix_sha256": attempt["prefix_sha256"],
        "attempt_prefix_last_index": attempt[
            "prefix_last_event_index"
        ],
        "attempt_prefix_last_sha": attempt[
            "prefix_last_event_sha256"
        ],
        "manifest_attempt_prefix": attempt_prefix,
        "final_bindings_match": final_bindings_match,
        "archive_outer_binding": bindings["raw_archive"],
        "archive_manifest_self_hash": archive[
            "manifest_self_hash_valid"
        ],
        "archive_zip_member_count": archive["zip_member_count"],
        "archive_duplicate_names": archive["duplicate_member_names"],
        "archive_name_set_exact": archive["member_name_set_exact"],
        "archive_bad_crc_member": archive["bad_crc_member"],
        "archive_entry_count": archive["entry_count"],
        "archive_unique_entries": archive["unique_entry_ids"],
        "archive_entry_rows": archive["entry_rows"],
        "archive_member_hash_failures": archive[
            "member_hash_failures"
        ],
        "archive_member_size_failures": archive[
            "member_size_failures"
        ],
        "archive_attempt_binding_failures": archive[
            "attempt_binding_failures"
        ],
        "archive_mapping_hash_valid": archive["mapping_hash_valid"],
        "archive_manifest_chunk_count": archive_manifest.get(
            "chunk_count"
        ),
        "archive_manifest_row_count": archive_manifest.get("row_count"),
        "gate_count": gate["row_count"],
        "gate_unique": gate["unique_gate_ids"],
        "gate_passed": gate["passed_count"],
        "gate_failed": gate["failed_count"],
        "gate_family_totals": gate["family_totals"],
        "gate_family_failures": gate["family_failures"],
        "gate_arithmetic_failures": gate["arithmetic_failures"],
        "gate_declared_pass_failures": gate[
            "declared_pass_failures"
        ],
        "gate_nonfinite_rows": gate["nonfinite_rows"],
        "gate_no_postselection": gate[
            "no_postselection_denominators"
        ],
        "gate_blueprint_matches": _gate_blueprint_matches(
            gate["rows"], blueprint
        ),
        "blueprint_count": len(blueprint),
        "blueprint_hash_matches": (
            _sha(blueprint)
            == config.get("gate_blueprint", {}).get(
                "canonical_blueprint_sha256"
            )
        ),
        "report_gate_summary": qualification.get("gate_summary"),
        "report_family_totals": report_family_totals,
        "report_family_failures": report_family_failures,
        "failed_ids_match": gate["failed_ids"]
        == qualification.get("failed_gate_ids"),
        "config_no_postselection": config.get("formal_matrix", {}).get(
            "no_postselection"
        ),
        "report_no_postselection": qualification.get(
            "statistical_procedure", {}
        ).get("postselection"),
        "manifest_counts": (
            manifest.get("expected_cells"),
            manifest.get("observed_cells"),
            manifest.get("expected_rows"),
            manifest.get("observed_rows"),
            manifest.get("exception_rows"),
        ),
        "manifest_status": manifest.get("status"),
        "qualification_verdict": qualification.get("verdict"),
        "verification_verdict": verification.get("verdict"),
        "release_verdict": release.get("verdict"),
        "release_pin_verdict": release_pin.get("verdict"),
        "qualified_claims": (
            qualification.get("qualified_claim"),
            verification.get("qualified_claim"),
            release.get("qualified_claim"),
            release_pin.get("qualified_claim"),
        ),
        "claims_all_null": all(
            _claims_literal_null(document)
            for document in (
                manifest,
                qualification,
                verification,
                release,
                release_pin,
            )
        ),
        "release_all_false": _release_all_false(release),
        "release_pin_all_false": _release_all_false(release_pin),
        "release_pin_byte_identical": (
            bindings["release"]["bytes"]
            == bindings["release_pin"]["bytes"]
            and bindings["release"]["sha256"]
            == bindings["release_pin"]["sha256"]
        ),
        "release_qualification_analysis_null": (
            release.get("qualification_analysis_sha256") is None
            and release_pin.get("qualification_analysis_sha256") is None
        ),
        "verification_qualification_analysis": verification.get(
            "qualification_analysis_sha256"
        ),
        "qualification_analysis": qualification.get("analysis_sha256"),
        "historical_flags": (
            qualification.get("historical_t9_2_4_no_go_preserved"),
            verification.get("historical_t9_2_4_no_go_preserved"),
            release.get("historical_t9_2_4_no_go_preserved"),
            release_pin.get("historical_t9_2_4_no_go_preserved"),
        ),
        "historical_lineage_binding": _path_sha_binding_matches(
            root,
            historical_binding,
            expected_path=EXPECTED_ARTIFACTS["historical_report"],
        ),
        "historical_verdict": historical.get("verdict"),
        "historical_qualified_claim": historical.get("qualified_claim"),
        "historical_claims_null": _historical_claims_literal_null(
            historical
        ),
        "old_formal_cell_data_accessed": qualification.get(
            "old_formal_cell_data_accessed"
        ),
        "runner_or_physics_imported": qualification.get(
            "runner_or_physics_imported"
        ),
        "semantic_rows_checked": semantic.get("rows_checked"),
        "semantic_max_error": max(
            float(value)
            for key, value in semantic.items()
            if key != "rows_checked"
        ),
        "raw_log_evidence_primary": qualification.get(
            "raw_log_evidence_diagnostic", {}
        ).get("primary"),
        "source_gate_byte_identical": (
            bindings["gate_ledger"]["bytes"]
            == bindings["qualification_source"]["bytes"]
            and bindings["gate_ledger"]["sha256"]
            == bindings["qualification_source"]["sha256"]
        ),
        "lfs_tracking": lfs,
        "large_ledger_bytes": bindings["cell_ledger"]["bytes"],
        "large_archive_bytes": bindings["raw_archive"]["bytes"],
        "bindings": bindings,
        "failed_gate_rows": [
            row for row in gate["rows"] if row["passed"] is False
        ],
    }


def audit_snapshot(snapshot: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "G01_task_and_schema_identity": snapshot.get("task_ids_exact") is True,
        "G02_manifest_execution_self_hash": snapshot.get(
            "manifest_self_hash"
        )
        is True,
        "G03_report_verification_self_hash": (
            snapshot.get("qualification_self_hash") is True
            and snapshot.get("verification_self_hash") is True
        ),
        "G04_release_and_lineage_self_hash": (
            snapshot.get("release_self_hash") is True
            and snapshot.get("release_pin_self_hash") is True
            and snapshot.get("lineage_self_hash") is True
            and snapshot.get("historical_self_hash") is True
        ),
        "G05_report_byte_bindings_live": snapshot.get(
            "report_bindings_match"
        )
        is True,
        "G06_verification_byte_bindings_live": snapshot.get(
            "verification_bindings_match"
        )
        is True,
        "G07_manifest_input_bindings_live": all(
            snapshot.get(key) is True
            for key in (
                "manifest_config_binding",
                "manifest_seal_binding",
                "manifest_ledger_binding",
                "manifest_archive_binding",
            )
        ),
        "G08_complete_row_denominator": (
            snapshot.get("row_count")
            == snapshot.get("row_unique")
            == 528_384
            and snapshot.get("manifest_counts")
            == (592, 592, 528_384, 528_384, 0)
        ),
        "G09_ledger_binding_recomputed": (
            snapshot.get("row_bytes")
            == snapshot.get("bindings", {}).get("cell_ledger", {}).get(
                "bytes"
            )
            and snapshot.get("row_sha256")
            == snapshot.get("bindings", {}).get("cell_ledger", {}).get(
                "sha256"
            )
        ),
        "G10_zero_exception_rows": (
            snapshot.get("ledger_exception_rows") == 0
            and snapshot.get("attempt_chunk_exceptions") == 0
        ),
        "G11_all_rows_conserve_and_finite": (
            snapshot.get("ledger_conservation_rows") == 528_384
            and snapshot.get("ledger_mandatory_nonfinite") == 0
        ),
        "G12_ledger_header_independently_bound": snapshot.get(
            "ledger_header_sha256"
        )
        == "5c1e4cf045ec84c6a3fe07bd23a496a80ace643e57d26dee6fb7885ef1b312d2",
        "G13_attempt_hash_chain_complete": (
            snapshot.get("attempt_chain_valid") is True
            and snapshot.get("attempt_indexes_valid") is True
            and snapshot.get("attempt_event_count") == 594
            and snapshot.get("attempt_first_kind") == "RUN_STARTED"
            and snapshot.get("attempt_final_kind") == "FINALIZED"
            and snapshot.get("attempt_run_errors") == 0
        ),
        "G14_attempt_chunks_complete": (
            snapshot.get("attempt_chunk_count") == 592
            and snapshot.get("attempt_chunk_rows") == 528_384
            and snapshot.get("attempt_chunks_complete") is True
            and snapshot.get("chunk_counts_match") is True
            and snapshot.get("ledger_chunk_count") == 592
        ),
        "G15_attempt_prefix_manifest_binding": (
            snapshot.get("manifest_attempt_prefix")
            == {
                "path": EXPECTED_ARTIFACTS["attempt_ledger"],
                "bytes": snapshot.get("attempt_prefix_bytes"),
                "sha256": snapshot.get("attempt_prefix_sha256"),
                "last_event_index": snapshot.get(
                    "attempt_prefix_last_index"
                ),
                "last_event_sha256": snapshot.get(
                    "attempt_prefix_last_sha"
                ),
            }
        ),
        "G16_final_event_binds_outputs": snapshot.get(
            "final_bindings_match"
        )
        is True,
        "G17_archive_manifest_and_crc": (
            snapshot.get("archive_manifest_self_hash") is True
            and snapshot.get("archive_bad_crc_member") is None
            and snapshot.get("archive_duplicate_names") == 0
            and snapshot.get("archive_name_set_exact") is True
        ),
        "G18_archive_chunk_and_row_accounting": (
            snapshot.get("archive_zip_member_count") == 594
            and snapshot.get("archive_entry_count")
            == snapshot.get("archive_unique_entries")
            == snapshot.get("archive_manifest_chunk_count")
            == 592
            and snapshot.get("archive_entry_rows")
            == snapshot.get("archive_manifest_row_count")
            == 528_384
        ),
        "G19_every_archived_chunk_hash_live": (
            snapshot.get("archive_member_hash_failures") == []
            and snapshot.get("archive_member_size_failures") == []
            and snapshot.get("archive_attempt_binding_failures") == []
            and snapshot.get("archive_mapping_hash_valid") is True
        ),
        "G20_gate_ledger_complete_unique": (
            snapshot.get("gate_count")
            == snapshot.get("gate_unique")
            == snapshot.get("blueprint_count")
            == 1_589
        ),
        "G21_gate_blueprint_exact": (
            snapshot.get("gate_blueprint_matches") is True
            and snapshot.get("blueprint_hash_matches") is True
        ),
        "G22_gate_bound_and_direction_recomputed": (
            snapshot.get("gate_arithmetic_failures") == []
            and snapshot.get("gate_declared_pass_failures") == []
            and snapshot.get("gate_nonfinite_rows") == []
        ),
        "G23_exact_gate_outcome": (
            snapshot.get("gate_passed") == 1_562
            and snapshot.get("gate_failed") == 27
            and snapshot.get("report_gate_summary")
            == {
                "all_passed": False,
                "failed": 27,
                "passed": 1_562,
                "total": 1_589,
            }
        ),
        "G24_family_totals_exact": (
            snapshot.get("gate_family_totals")
            == EXPECTED_FAMILY_TOTALS
            and snapshot.get("report_family_totals")
            == EXPECTED_FAMILY_TOTALS
        ),
        "G25_failure_family_counts_exact": (
            snapshot.get("gate_family_failures")
            == EXPECTED_FAILED_FAMILIES
            and snapshot.get("report_family_failures")
            == EXPECTED_FAILED_FAMILIES
        ),
        "G26_failed_gate_ids_exact": snapshot.get("failed_ids_match") is True,
        "G27_no_postselection": (
            snapshot.get("gate_no_postselection") is True
            and snapshot.get("config_no_postselection") is True
            and snapshot.get("report_no_postselection") is False
        ),
        "G28_raw_log_evidence_diagnostic_only": snapshot.get(
            "raw_log_evidence_primary"
        )
        is False,
        "G29_scientific_no_go_not_incomplete": (
            snapshot.get("manifest_status")
            == "FORMAL_RAW_EVIDENCE_COMPLETE"
            and snapshot.get("qualification_verdict")
            == snapshot.get("verification_verdict")
            == snapshot.get("release_verdict")
            == snapshot.get("release_pin_verdict")
            == FORMAL_NO_GO
        ),
        "G30_no_go_releases_nothing": (
            snapshot.get("release_all_false") is True
            and snapshot.get("release_pin_all_false") is True
        ),
        "G31_all_claims_literal_null": snapshot.get("claims_all_null") is True,
        "G32_qualified_claim_literal_null": snapshot.get(
            "qualified_claims"
        )
        == (None, None, None, None),
        "G33_no_go_release_analysis_null": snapshot.get(
            "release_qualification_analysis_null"
        )
        is True,
        "G34_verification_binds_qualification_analysis": snapshot.get(
            "verification_qualification_analysis"
        )
        == snapshot.get("qualification_analysis"),
        "G35_release_pin_byte_identical": snapshot.get(
            "release_pin_byte_identical"
        )
        is True,
        "G36_historical_no_go_preserved": (
            snapshot.get("historical_flags") == (True, True, True, True)
            and snapshot.get("historical_lineage_binding") is True
            and snapshot.get("historical_verdict")
            == "NO_GO_TWIN_QUALIFICATION"
            and snapshot.get("historical_qualified_claim") is None
            and snapshot.get("historical_claims_null") is True
        ),
        "G37_no_old_cell_or_runtime_import": (
            snapshot.get("old_formal_cell_data_accessed") is False
            and snapshot.get("runner_or_physics_imported") is False
        ),
        "G38_independent_semantic_receipt_complete": (
            snapshot.get("semantic_rows_checked") == 528_384
            and isinstance(snapshot.get("semantic_max_error"), float)
            and snapshot["semantic_max_error"] <= 2e-14
        ),
        "G39_gate_source_copy_byte_identical": snapshot.get(
            "source_gate_byte_identical"
        )
        is True,
        "G40_large_evidence_lfs_tracked": (
            snapshot.get("lfs_tracking", {}).get("all_tracked") is True
            and snapshot.get("large_ledger_bytes", 0) > 100_000_000
            and snapshot.get("large_archive_bytes", 0) > 100_000_000
        ),
    }


def _snapshot_mutations() -> tuple[tuple[str, str, object], ...]:
    return (
        ("M01_manifest_hash", "manifest_self_hash", False),
        ("M02_report_hash", "qualification_self_hash", False),
        ("M03_release_hash", "release_self_hash", False),
        ("M04_binding_tamper", "report_bindings_match", False),
        ("M05_drop_row", "row_count", 528_383),
        ("M06_duplicate_row", "row_unique", 528_383),
        ("M07_hide_exception", "ledger_exception_rows", 1),
        ("M08_conservation_failure", "ledger_conservation_rows", 528_383),
        ("M09_nan_row", "ledger_mandatory_nonfinite", 1),
        ("M10_header_drift", "ledger_header_sha256", "0" * 64),
        ("M11_attempt_chain", "attempt_chain_valid", False),
        ("M12_attempt_index", "attempt_indexes_valid", False),
        ("M13_run_error", "attempt_run_errors", 1),
        ("M14_drop_chunk", "attempt_chunk_count", 591),
        ("M15_chunk_rows", "attempt_chunk_rows", 528_383),
        ("M16_prefix_hash", "attempt_prefix_sha256", "0" * 64),
        ("M17_final_binding", "final_bindings_match", False),
        ("M18_archive_crc", "archive_bad_crc_member", "chunk.npz"),
        ("M19_archive_duplicate", "archive_duplicate_names", 1),
        ("M20_archive_drop", "archive_entry_count", 591),
        ("M21_archive_row", "archive_entry_rows", 528_383),
        ("M22_archive_member_hash", "archive_member_hash_failures", ["x"]),
        ("M23_archive_size", "archive_member_size_failures", ["x"]),
        ("M24_archive_attempt", "archive_attempt_binding_failures", ["x"]),
        ("M25_archive_mapping", "archive_mapping_hash_valid", False),
        ("M26_drop_gate", "gate_count", 1_588),
        ("M27_duplicate_gate", "gate_unique", 1_588),
        ("M28_blueprint_switch", "gate_blueprint_matches", False),
        ("M29_blueprint_hash", "blueprint_hash_matches", False),
        ("M30_bound_factor", "gate_arithmetic_failures", ["g"]),
        ("M31_bound_direction", "gate_declared_pass_failures", ["g"]),
        ("M32_nonfinite_gate", "gate_nonfinite_rows", ["g"]),
        ("M33_promote_gate", "gate_failed", 26),
        ("M34_family_total", "gate_family_totals", {}),
        ("M35_failure_family", "gate_family_failures", {}),
        ("M36_failed_id", "failed_ids_match", False),
        ("M37_postselection", "config_no_postselection", False),
        ("M38_raw_score_primary", "raw_log_evidence_primary", True),
        ("M39_promote_verdict", "qualification_verdict", FORMAL_PASS),
        ("M40_incomplete_verdict", "verification_verdict", FORMAL_INCOMPLETE),
        ("M41_release_task", "release_all_false", False),
        ("M42_promote_claim", "claims_all_null", False),
        ("M43_qualified_claim", "qualified_claims", (QUALIFIED_CLAIM,) * 4),
        ("M44_release_analysis", "release_qualification_analysis_null", False),
        ("M45_release_pin", "release_pin_byte_identical", False),
        ("M46_rewrite_old_no_go", "historical_verdict", FORMAL_PASS),
        ("M47_old_cell_access", "old_formal_cell_data_accessed", True),
        ("M48_import_physics", "runner_or_physics_imported", True),
        ("M49_semantic_drop", "semantic_rows_checked", 528_383),
        ("M50_source_copy", "source_gate_byte_identical", False),
        ("M51_lfs_removed", "lfs_tracking", {"all_tracked": False}),
    )


def _branch_mutations() -> list[dict[str, object]]:
    cases = (
        ("B01_pass_claim_null", "pass", "qualified_claim", None),
        ("B02_pass_partial_release", "pass", "released", {}),
        ("B03_pass_gate_fail", "pass", "all_gates_passed", False),
        ("B04_pass_incomplete", "pass", "complete", False),
        ("B05_no_go_claim", "no_go", "qualified_claim", QUALIFIED_CLAIM),
        ("B06_no_go_release", "no_go", "released", {task: True for task in DOWNSTREAM_TASKS}),
        ("B07_no_go_all_pass", "no_go", "all_gates_passed", True),
        ("B08_no_go_incomplete", "no_go", "complete", False),
        ("B09_incomplete_claim", "incomplete", "qualified_claim", QUALIFIED_CLAIM),
        ("B10_incomplete_release", "incomplete", "released", {task: True for task in DOWNSTREAM_TASKS}),
        ("B11_incomplete_complete", "incomplete", "complete", True),
        ("B12_unknown_verdict", "pass", "verdict", "PASS"),
        ("B13_claim_false", "pass", "claim_state", {**{field: None for field in CLAIM_FIELDS}, "rank": False}),
        ("B14_claim_missing", "no_go", "claim_state", {}),
        ("B15_claim_extra", "incomplete", "claim_state", {**{field: None for field in CLAIM_FIELDS}, "surpass": None}),
        ("B16_rewrite_old", "pass", "historical_no_go_preserved", False),
        ("B17_no_go_rewrite_old", "no_go", "historical_no_go_preserved", False),
        ("B18_incomplete_rewrite_old", "incomplete", "historical_no_go_preserved", False),
    )
    results: list[dict[str, object]] = []
    for mutation_id, branch, field, replacement in cases:
        fixture = _branch_fixture(branch)
        fixture[field] = replacement
        results.append(
            {
                "mutation_id": mutation_id,
                "branch": branch,
                "field": field,
                "detected": not all(
                    audit_branch_fixture(fixture).values()
                ),
            }
        )
    return results


def run_mutation_audit(
    snapshot: Mapping[str, Any],
) -> list[dict[str, object]]:
    baseline = audit_snapshot(snapshot)
    if not all(baseline.values()):
        failed = [name for name, passed in baseline.items() if not passed]
        raise ValueError(f"post-outcome baseline failed: {failed}")
    results: list[dict[str, object]] = []
    for mutation_id, field, replacement in _snapshot_mutations():
        mutated = copy.deepcopy(dict(snapshot))
        mutated[field] = replacement
        gates = audit_snapshot(mutated)
        results.append(
            {
                "mutation_id": mutation_id,
                "field": field,
                "detected": not all(gates.values()),
                "failed_gates": [
                    name for name, passed in gates.items() if not passed
                ],
            }
        )
    return results


def _source_rows(
    gates: Mapping[str, bool],
    mutations: Sequence[Mapping[str, object]],
    branch_mutations: Sequence[Mapping[str, object]],
    snapshot: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gate_id, passed in gates.items():
        rows.append(
            {
                "record_type": "audit_gate",
                "record_id": gate_id,
                "family": "",
                "metric": "",
                "passed": passed,
                "detail": "",
            }
        )
    for row in mutations:
        rows.append(
            {
                "record_type": "semantic_mutation",
                "record_id": row["mutation_id"],
                "family": "",
                "metric": str(row["field"]),
                "passed": row["detected"],
                "detail": "|".join(row["failed_gates"]),
            }
        )
    for row in branch_mutations:
        rows.append(
            {
                "record_type": "branch_mutation",
                "record_id": row["mutation_id"],
                "family": str(row["branch"]),
                "metric": str(row["field"]),
                "passed": row["detected"],
                "detail": "",
            }
        )
    for row in snapshot["failed_gate_rows"]:
        rows.append(
            {
                "record_type": "failed_formal_gate",
                "record_id": row["gate_id"],
                "family": row["family"],
                "metric": row["metric"],
                "passed": False,
                "detail": f"direction={row['direction']};margin={row['margin']}",
            }
        )
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    fields = (
        "record_type",
        "record_id",
        "family",
        "metric",
        "passed",
        "detail",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def build_audit(
    root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    base = (root or _root()).resolve()
    snapshot = build_snapshot(base)
    gates = audit_snapshot(snapshot)
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"post-outcome audit gates failed: {failed}")
    mutations = run_mutation_audit(snapshot)
    branch_mutations = _branch_mutations()
    if not all(row["detected"] is True for row in mutations):
        raise ValueError("post-outcome semantic mutation escaped")
    if not all(row["detected"] is True for row in branch_mutations):
        raise ValueError("post-outcome branch mutation escaped")
    rows = _source_rows(gates, mutations, branch_mutations, snapshot)
    source_payload = _csv_bytes(rows)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "verdict": VERDICT,
        "formal_verdict": FORMAL_NO_GO,
        "qualified_claim": None,
        "historical_t9_2_4_no_go_preserved": True,
        "evidence_bindings": snapshot["bindings"],
        "source_data": {
            "path": SOURCE_PATH,
            "bytes": len(source_payload),
            "sha256": _sha_bytes(source_payload),
            "rows": len(rows),
        },
        "execution_summary": {
            "rows": 528_384,
            "chunks": 592,
            "attempt_events": 594,
            "exception_rows": 0,
            "archive_members": 594,
        },
        "gate_outcome": {
            "total": 1_589,
            "passed": 1_562,
            "failed": 27,
            "family_totals": EXPECTED_FAMILY_TOTALS,
            "failed_family_counts": EXPECTED_FAILED_FAMILIES,
            "failed_gate_ids": [
                row["gate_id"] for row in snapshot["failed_gate_rows"]
            ],
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(value is True for value in gates.values()),
            "total": len(gates),
            "all_passed": all(gates.values()),
        },
        "semantic_mutations": mutations,
        "branch_mutations": branch_mutations,
        "mutation_summary": {
            "detected": sum(
                row["detected"] is True
                for row in [*mutations, *branch_mutations]
            ),
            "total": len(mutations) + len(branch_mutations),
            "all_detected": all(
                row["detected"] is True
                for row in [*mutations, *branch_mutations]
            ),
        },
        "claim_state": {field: None for field in CLAIM_FIELDS},
        "release_state": {task: False for task in DOWNSTREAM_TASKS},
        "claim_boundary": (
            "The audit confirms a complete fresh synthetic-task scientific "
            "NO-GO. It qualifies no physical accuracy, LER, lifetime, "
            "break-even, Puviani, external-SOTA, hardware, or rank claim."
        ),
    }
    report["analysis_sha256"] = _sha(report)
    return report, rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_artifacts(
    root: Path | None = None,
) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report, rows = build_audit(base)
    source_payload = _csv_bytes(rows)
    if _sha_bytes(source_payload) != report["source_data"]["sha256"]:
        raise RuntimeError("source-data serialization drift")
    _atomic_write(base / SOURCE_PATH, source_payload)
    _atomic_write(
        base / OUTPUT_PATH,
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_root())
    args = parser.parse_args(argv)
    report = write_artifacts(args.root)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "gate_summary": report["gate_summary"],
                "mutation_summary": report["mutation_summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_FIELDS",
    "DOWNSTREAM_TASKS",
    "EXPECTED_FAILED_FAMILIES",
    "EXPECTED_FAMILY_TOTALS",
    "FORMAL_INCOMPLETE",
    "FORMAL_NO_GO",
    "FORMAL_PASS",
    "OUTPUT_PATH",
    "SCHEMA_VERSION",
    "SOURCE_PATH",
    "TASK_ID",
    "VERDICT",
    "audit_branch_fixture",
    "audit_snapshot",
    "build_audit",
    "build_snapshot",
    "run_mutation_audit",
    "write_artifacts",
]
