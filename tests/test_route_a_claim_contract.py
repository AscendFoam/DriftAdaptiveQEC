from __future__ import annotations

import csv
import hashlib
import json

from cnn_fpga.benchmark.route_a_claim_contract import (
    DEFAULT_CSV,
    DEFAULT_JSON,
    ROOT,
    SUPPORTED,
    build_contract,
    load_evidence,
    validate_contract,
)


def _report() -> dict[str, object]:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def test_contract_recomputes_all_gates() -> None:
    report = _report()
    assert report["status"] == "PASS"
    assert report["verdict"] == "PASS_ROUTE_A_CLAIM_ROLE_AND_LANE_CONTRACT_FROZEN"
    assert report["gate_summary"] == {"passed": 20, "total": 20, "failed": []}
    assert all(report["gates"].values())
    assert all(validate_contract(build_contract(), load_evidence()).values())


def test_roles_have_one_primary_and_learning_is_replaceable() -> None:
    report = _report()
    roles = {row["role_id"]: row for row in report["roles"]}
    assert len(roles) == 11
    assert report["primary_system_role_id"] == "ROLE-SYSTEM"
    assert sum(row["role_type"] == "primary_system" for row in roles.values()) == 1
    for role_id in ("ROLE-CNN", "ROLE-TEACHER", "ROLE-STUDENT"):
        assert roles[role_id]["role_type"] == "replaceable_learning_extension"
        assert roles[role_id]["replaceable"] is True
    assert roles["ROLE-ORACLE"]["deployability"] == "nondeployable_hidden_truth"
    assert "never the logical action" in roles["ROLE-REGIME"]["output_authority"]
    assert "logical decision" in roles["ROLE-MAP"]["output_authority"]


def test_three_lane_metric_namespaces_are_disjoint_and_global_score_is_off() -> None:
    report = _report()
    lanes = report["comparison_lanes"]
    assert {row["lane_id"] for row in lanes} == {"LANE-DECODER", "LANE-GQF", "LANE-HARDWARE"}
    metric_sets = [set(row["allowed_metrics"]) for row in lanes]
    assert all(metric_sets[i].isdisjoint(metric_sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert all(value is False for value in report["cross_lane_rule"].values())


def test_claim_statuses_preserve_negative_and_evidence_boundaries() -> None:
    report = _report()
    claims = {row["claim_id"]: row for row in report["claims"]}
    assert len(claims) == 11
    assert claims["CLAIM-CNN-01"]["support_status"] == "ABLATION_ONLY"
    assert claims["CLAIM-STUDENT-01"]["support_status"] == "SUPPORTED_EXTENSION_ONLY"
    assert claims["CLAIM-STUDENT-01"]["lane_id"] == "LANE-GOVERNANCE"
    for claim_id in ("CLAIM-GQF-01", "CLAIM-HW-SPEED-01", "CLAIM-BREAK-EVEN-01"):
        assert claims[claim_id]["support_status"] == "PROHIBITED_NOW"
        assert claims[claim_id]["current_evidence"] == []
    assert claims["CLAIM-RTL-QUAL-01"]["support_status"] == "SUPPORTED_BOUNDED"
    assert "board-independent" in claims["CLAIM-RTL-QUAL-01"]["canonical_wording_en"]
    assert "not a board measurement" in claims["CLAIM-FPGA-EST-01"]["canonical_wording_en"]
    for row in claims.values():
        assert row["activation_gate"]
        assert row["revocation_conditions"]
        assert row["forbidden_wording"]
        if row["support_status"] in SUPPORTED:
            assert row["current_evidence"]


def test_semantic_mutations_and_source_bindings_are_current() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 10
    assert all(row["rejected"] for row in audit["mutations"])
    for binding in report["source_bindings"]:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]


def test_source_data_has_one_complete_row_per_claim() -> None:
    with DEFAULT_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 11
    assert len({row["claim_id"] for row in rows}) == 11
    assert all(row["canonical_wording_en"] and row["activation_gate"] and row["revocation_conditions"] for row in rows)
    report = _report()
    assert report["source_data"]["row_count"] == 11
    assert hashlib.sha256(DEFAULT_CSV.read_bytes()).hexdigest() == report["source_data"]["sha256"]

