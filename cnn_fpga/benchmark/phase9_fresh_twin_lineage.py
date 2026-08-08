"""Immutable lineage guard for the fresh Phase-9 twin qualification.

This module deliberately does not import either physics backend, the old
formal runner, or the old verifier.  It is the only fresh-transaction module
that may open historical T9.2.4 JSON governance artifacts.  Cell ledgers,
source data, and state archives are outside its allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "T-RISK-20260726-01"
SCHEMA_VERSION = "1.0"
PASS_VERDICT = "PASS_HISTORICAL_NO_GO_LINEAGE_BOUND"

HISTORICAL_NULL_FIELDS = (
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "hardware_measured",
    "rank",
)

HISTORICAL_BINDINGS: Mapping[str, Mapping[str, str]] = {
    "parent_preregistration": {
        "path": "docs/t9_2_4_twin_qualification_preregistration.json",
        "sha256": "144cef9f06937cb1388343c96b19499f32ed03dd1a27f44eeb8884cf4da42e01",
        "analysis_sha256": "98e72c457dab941daab270b3bd63eec939564a6f1fedaad061eab59280988695",
        "verdict": "PASS_T9_2_4_PREREGISTRATION_FROZEN",
    },
    "child_amendment": {
        "path": "docs/t9_2_4_formal_runner_amendment_seal.json",
        "sha256": "9f0977729d73b18fc4630403e4c3bea30d8da9ddc0ff7fb2765085061e34ca61",
        "analysis_sha256": "f2a2ab287c59e0947768cd08824856d7b9046b79c3d713b18043efe09de22944",
        "verdict": "",
    },
    "formal_report": {
        "path": "docs/t9_2_4_dual_backend_qualification.json",
        "sha256": "b81059cb685f76848f5fd978acf0cbdaf52c39a26588f975fa6e7c9b2378c1d1",
        "analysis_sha256": "d3f05e69db9c4799d55ce0813f8bd52cdb22384dbd5cfc68227715c44c53ec49",
        "verdict": "NO_GO_TWIN_QUALIFICATION",
    },
    "release_pin": {
        "path": "configs/phase9/t9_2_4_release_pin.json",
        "sha256": "d8e6cf522830447c6922d44ba46b275bf61ab67b7f5884e159eb4863b9fe77d3",
        "analysis_sha256": "d3f05e69db9c4799d55ce0813f8bd52cdb22384dbd5cfc68227715c44c53ec49",
        "verdict": "NO_GO_TWIN_QUALIFICATION",
    },
    "post_outcome_audit": {
        "path": "docs/t9_2_4_dual_backend_post_outcome_audit.json",
        "sha256": "cc580481c440ccafeee92d2e9f7db51310ad55db85fc971b243d8a71eeedb953",
        "analysis_sha256": "b9b0de857f8f345e653600091899f7d87f18d7706fc8909bc1f969d15bb6e6d3",
        "verdict": "AUDIT_CONFIRMS_SCIENTIFIC_NO_GO",
    },
}

# Constructed to keep prohibited historical cell-level names out of fresh
# source text as literal strings.  This module never opens these paths.
PROHIBITED_HISTORICAL_BASENAMES = (
    "t9_2_4_dual_backend_" + "cell_ledger.csv",
    "t9_2_4_dual_backend_" + "qualification_source_data.csv",
    "t9_2_4_dual_backend_" + "state_archive.npz",
)

FRESH_SOURCE_PREFIXES = (
    "cnn_fpga/benchmark/phase9_fresh_twin_",
    "cnn_fpga/benchmark/phase9_iq_semantics_diagnostic.py",
    "physics/phase9_iq_likelihood_reference.py",
    "configs/phase9/t_risk_20260726_01_",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )


def _strict_json_object(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _claim_state(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    state = document.get("claim_state")
    if state is None:
        state = document.get("current_null_state")
    if not isinstance(state, dict):
        raise ValueError(f"{name} claim state missing")
    if tuple(state.keys()) != HISTORICAL_NULL_FIELDS:
        raise ValueError(f"{name} historical claim schema drift")
    if any(value is not None for value in state.values()):
        raise ValueError(f"{name} historical claim is not literal null")
    return state


def _fresh_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(relative.startswith(prefix) for prefix in FRESH_SOURCE_PREFIXES):
            candidates.append(path)
    return sorted(candidates)


def _scan_fresh_sources(root: Path) -> dict[str, object]:
    scanned: list[str] = []
    violations: list[dict[str, str]] = []
    for path in _fresh_candidates(root):
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Generated binary outputs are never members of the source prefixes.
            violations.append({"path": relative, "reason": "non_utf8_source"})
            continue
        scanned.append(relative)
        for basename in PROHIBITED_HISTORICAL_BASENAMES:
            if basename in text:
                violations.append(
                    {"path": relative, "reason": f"prohibited_reference:{basename}"}
                )
    return {
        "scanned_paths": scanned,
        "prohibited_historical_basenames": list(PROHIBITED_HISTORICAL_BASENAMES),
        "violations": violations,
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    """Validate immutable historical governance inputs and build a receipt."""

    base = (root or _root()).resolve()
    bindings: dict[str, dict[str, object]] = {}
    for name, expected in HISTORICAL_BINDINGS.items():
        relative = expected["path"]
        path = (base / relative).resolve()
        if base not in path.parents:
            raise ValueError(f"{name} path escapes repository")
        payload = path.read_bytes()
        actual_sha = _sha_bytes(payload)
        if actual_sha != expected["sha256"]:
            raise ValueError(f"{name} byte hash mismatch")
        document = _strict_json_object(payload, name)
        if document.get("task_id") != "T9.2.4":
            raise ValueError(f"{name} task id mismatch")
        if document.get("analysis_sha256") != expected["analysis_sha256"]:
            raise ValueError(f"{name} analysis hash mismatch")
        expected_verdict = expected["verdict"]
        if expected_verdict and document.get("verdict") != expected_verdict:
            raise ValueError(f"{name} verdict mismatch")
        _claim_state(document, name)
        bindings[name] = {
            "path": relative,
            "sha256": actual_sha,
            "analysis_sha256": document["analysis_sha256"],
            "verdict": document.get("verdict"),
            "claim_fields": list(HISTORICAL_NULL_FIELDS),
            "all_claims_literal_null": True,
        }

    if (
        bindings["formal_report"]["analysis_sha256"]
        != bindings["release_pin"]["analysis_sha256"]
    ):
        raise ValueError("historical formal report/release analysis mismatch")

    scan = _scan_fresh_sources(base)
    gates = {
        "G01_parent_preregistration_byte_bound": True,
        "G02_child_amendment_byte_bound": True,
        "G03_formal_no_go_byte_bound": True,
        "G04_release_pin_byte_bound": True,
        "G05_post_outcome_audit_byte_bound": True,
        "G06_formal_release_analysis_identical": True,
        "G07_historical_verdict_no_go": (
            bindings["formal_report"]["verdict"] == "NO_GO_TWIN_QUALIFICATION"
        ),
        "G08_historical_claims_literal_null": all(
            item["all_claims_literal_null"] for item in bindings.values()
        ),
        "G09_historical_parent_rewritten_false": True,
        "G10_no_fresh_cell_level_historical_access": not scan["violations"],
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "purpose": "immutable_historical_no_go_lineage_only",
        "historical_parent_rewritten": False,
        "historical_bindings": bindings,
        "fresh_source_scan": scan,
        "gates": gates,
        "gate_summary": {
            "passed": sum(value is True for value in gates.values()),
            "total": len(gates),
        },
        "verdict": PASS_VERDICT if all(gates.values()) else "INCOMPLETE_FAIL_CLOSED",
    }
    report["analysis_sha256"] = _sha_bytes(_canonical(report))
    return report


def verify_receipt(receipt: Mapping[str, Any], root: Path | None = None) -> bool:
    """Rebuild the receipt live; reject missing, extra, or stale content."""

    rebuilt = build_receipt(root)
    return _canonical(dict(receipt)) == _canonical(rebuilt)


def write_receipt(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Atomically write the deterministic historical-lineage receipt."""

    report = build_receipt(root)
    if report["verdict"] != PASS_VERDICT:
        raise RuntimeError("historical lineage did not pass")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the immutable historical NO-GO lineage receipt."
    )
    parser.parse_args(argv)
    destination = _root() / "docs/t_risk_20260726_01_historical_no_go_receipt.json"
    report = write_receipt(destination)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "path": destination.relative_to(_root()).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
