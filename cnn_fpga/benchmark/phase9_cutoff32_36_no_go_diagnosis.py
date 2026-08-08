"""Machine-checkable diagnosis of the cutoff32/36 design NO-GO.

The diagnosis consumes only the independently verified V2 publication.  It
classifies observed failures without changing any completed gate, margin or
verdict, and freezes the bounded next repair direction: one fresh 36/40/44
transaction plus a Rao-Blackwellized RESET qualification estimand.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark import (
    phase9_cutoff32_36_design_diagnostic_v2_verify as verifier,
)


TASK_ID = "T-RISK-20260728-01"
OUTPUT_PATH = "docs/t_risk_20260728_01_cutoff32_36_no_go_diagnosis.json"
MANIFEST_PATH = (
    "docs/t_risk_20260728_01_cutoff32_36_design_extension_fresh2_manifest.json"
)
DIAGNOSIS_VERDICT = "PHYSICS_AND_ESTIMAND_REPAIR_REQUIRED"
CLAIM_BOUNDARY = dict(verifier.CLAIM_BOUNDARY)


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


def _load_shared_rows(
    root: Path, manifest: Mapping[str, Any]
) -> dict[tuple[int, str], list[dict[str, str]]]:
    output: dict[tuple[int, str], list[dict[str, str]]] = {}
    for receipt in manifest["chunk_receipts"]:
        cell = receipt["cell"]
        if cell["layer"] != "shared":
            continue
        rows = list(
            csv.DictReader(
                (root / receipt["csv"]["path"]).open(encoding="utf-8", newline="")
            )
        )
        key = (int(cell["cutoff"]), str(cell["backend"]))
        if (
            key in output
            or len(rows) != 72
            or any(row["terminal_round"] != "True" for row in rows)
        ):
            raise RuntimeError("shared diagnosis denominator drift")
        output[key] = rows
    expected = {(cutoff, backend) for cutoff in (28, 32, 36) for backend in ("A", "B")}
    if set(output) != expected:
        raise RuntimeError("shared diagnosis coverage drift")
    return output


def diagnose(root: Path | None = None) -> dict[str, Any]:
    repository = _root() if root is None else root.resolve()
    verification = verifier.verify(repository)
    if verification["verification_verdict"] != verifier.VERIFIED_VERDICT:
        raise RuntimeError("diagnosis requires independently verified NO-GO")
    report_path = repository / verifier.REPORT_PATH
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = [row for row in report["gate_rows"] if row["passed"] is False]
    failed_28_32 = [row for row in failed if row["cutoff_or_increment"] == "28->32"]
    passed_32_36 = [
        row
        for row in report["gate_rows"]
        if row["cutoff_or_increment"] == "32->36" and row["passed"] is True
    ]
    failed_32_36 = [
        row
        for row in report["gate_rows"]
        if row["cutoff_or_increment"] == "32->36" and row["passed"] is False
    ]
    tail_failures = [row for row in failed if row["family"] == "fault_absolute_tail"]
    shared_failures = [row for row in failed if row["family"] == "shared_scalar"]
    if (
        len(failed) != 61
        or len(failed_28_32) != 56
        or len(passed_32_36) != 596
        or failed_32_36
        or [row["gate_id"] for row in tail_failures]
        != [
            "fault/tail/commutator_defect/c36/step/A/+",
            "fault/tail/commutator_defect/c36/step/A/+i",
        ]
        or len(shared_failures) != 3
        or {row["cutoff_or_increment"] for row in shared_failures} != {"28", "32", "36"}
        or any(
            row["metric"] != "level_probability_l1"
            or row["backend_or_pair"] != "A/B"
            or row["estimate"] != 0.1388888888888889
            for row in shared_failures
        )
    ):
        raise RuntimeError("NO-GO failure decomposition drift")
    manifest_path = repository / MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shared = _load_shared_rows(repository, manifest)
    reset_audits: dict[str, dict[str, object]] = {}
    for cutoff in (28, 32, 36):
        by_backend: dict[str, dict[str, float | int]] = {}
        for backend in ("A", "B"):
            rows = shared[(cutoff, backend)]
            successes = sum(row["reset_hidden_success"] == "True" for row in rows)
            by_backend[backend] = {
                "sample_count": len(rows),
                "sampled_success_count": successes,
                "sampled_success_rate": successes / len(rows),
                "rao_blackwell_success_mean": mean(
                    float(row["rao_blackwell_reset_success"]) for row in rows
                ),
                "post_reset_level_g_mean": mean(float(row["level_g"]) for row in rows),
            }
        sampled_l1 = 2.0 * abs(
            float(by_backend["A"]["post_reset_level_g_mean"])
            - float(by_backend["B"]["post_reset_level_g_mean"])
        )
        rb_difference = abs(
            float(by_backend["A"]["rao_blackwell_success_mean"])
            - float(by_backend["B"]["rao_blackwell_success_mean"])
        )
        if (
            by_backend["A"]["sampled_success_count"] != 59
            or by_backend["B"]["sampled_success_count"] != 54
            or sampled_l1 != 0.13888888888888884
            or rb_difference > 0.003
        ):
            raise RuntimeError("shared RESET diagnosis drift")
        reset_audits[str(cutoff)] = {
            "backends": by_backend,
            "sampled_post_reset_level_l1": sampled_l1,
            "rao_blackwell_success_absolute_difference": rb_difference,
            "classification": (
                "INDEPENDENT_BERNOULLI_BRANCH_NOISE_CONTAMINATES_"
                "QUALIFICATION_ESTIMAND"
            ),
        }
    diagnosis: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": ("PHASE9-CUTOFF32-36-NO-GO-DIAGNOSIS-V1"),
        "status": "COMPLETE",
        "diagnosis_verdict": DIAGNOSIS_VERDICT,
        "scientific_verdict_unchanged": verifier.NO_GO_VERDICT,
        "failure_decomposition": {
            "total_failed_gates": len(failed),
            "cutoff_28_to_32_failed_gates": len(failed_28_32),
            "cutoff_32_to_36_passed_gates": len(passed_32_36),
            "cutoff_32_to_36_failed_gates": len(failed_32_36),
            "cutoff36_fault_absolute_tail_failed_gate_ids": [
                row["gate_id"] for row in tail_failures
            ],
            "shared_reset_failed_gate_ids": [row["gate_id"] for row in shared_failures],
            "maximum_margin_ratio": report["maximum_margin_ratio"],
        },
        "root_causes": {
            "cutoff28": (
                "PHYSICAL_TRUNCATION_NOT_CONVERGED; seed extension and "
                "logical-survival-only checks cannot repair it"
            ),
            "cutoff36": (
                "RESIDUAL_STATE_CONDITIONED_BOUNDARY_TAIL; two step/A "
                "logical states violate the commutator-defect gate"
            ),
            "shared_reset": (
                "QUALIFICATION_ESTIMAND_USES_STOCHASTIC_POST_RESET_BRANCH; "
                "independent backend Bernoulli draws dominate the A/B point"
            ),
        },
        "shared_reset_audit": reset_audits,
        "bounded_repair_preregistration": {
            "new_task_id": "T-RISK-20260728-03",
            "fresh_cutoffs": [36, 40, 44],
            "required_consecutive_increments": [[36, 40], [40, 44]],
            "absolute_tail_cutoff": 44,
            "adapter_cap": 44,
            "resource_preflight_before_scientific_rows": True,
            "shared_reset_primary_estimand": (
                "RAO_BLACKWELLIZED_EXPECTED_POST_RESET_DENSITY_AND_LEVELS"
            ),
            "stochastic_reset_role": "stress_only_nonvoting",
            "reuse_old_passing_gates": False,
            "reuse_old_raw_rows_for_new_decision": False,
            "terminal_if_cutoff44_fails": True,
            "automatic_cutoff_extension_beyond_44": False,
            "powered_formal_release": False,
        },
        "bindings": {
            "verified_no_go": _binding(
                repository / verifier.VERIFICATION_PATH, repository
            ),
            "report": _binding(report_path, repository),
            "manifest": _binding(manifest_path, repository),
            "diagnosis_source": _binding(Path(__file__).resolve(), repository),
        },
        "qualified_claim": None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    diagnosis["analysis_sha256"] = _sha(diagnosis)
    return diagnosis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose the independently verified cutoff NO-GO."
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    repository = _root()
    result = diagnose(repository)
    output = (repository / args.output).resolve()
    output.relative_to(repository)
    output.write_bytes(
        (
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )
    print(
        json.dumps(
            {
                "diagnosis_verdict": result["diagnosis_verdict"],
                "failed_gates": result["failure_decomposition"]["total_failed_gates"],
                "cutoff_32_to_36_failed": result["failure_decomposition"][
                    "cutoff_32_to_36_failed_gates"
                ],
                "next_task": result["bounded_repair_preregistration"]["new_task_id"],
                "analysis_sha256": result["analysis_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
