from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "claim_ladder.json"
HUMAN_PATH = ROOT / "docs" / "claim_ladder.md"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_claim_ladder_has_exact_five_level_contract():
    contract = _contract()
    levels = contract["levels"]

    assert contract["schema_version"] == "claim-ladder-v1"
    assert contract["level_order"] == ["CL1", "CL2", "CL3", "CL4", "CL5"]
    assert [level["id"] for level in levels] == contract["level_order"]
    assert [level["canonical_name"] for level in levels] == [
        "simulation",
        "synthesis_estimate",
        "board_measurement",
        "board_hil_replay",
        "quantum_experiment",
    ]
    assert contract["current_strongest_project_level"] == "CL1"


def test_each_level_has_wording_boundary_gate_and_existing_anchors():
    for level in _contract()["levels"]:
        assert level["allowed_wording"]
        assert level["forbidden_wording"]
        assert len(level["promotion_gate"]) >= 3
        assert level["evidence_paths"]
        for relative_path in level["evidence_paths"]:
            assert (ROOT / relative_path).exists(), relative_path


def test_current_status_cannot_silently_promote_hardware_claims():
    statuses = {level["id"]: level["current_status"] for level in _contract()["levels"]}

    assert statuses["CL1"] == "supported_bounded"
    assert statuses["CL2"] == "blocked_missing_synthesis_report"
    assert statuses["CL3"] == "blocked_missing_board_host_execution"
    assert statuses["CL4"] == "blocked_no_board_hil"
    assert statuses["CL5"] == "blocked_no_quantum_experiment"


def test_orthogonal_runtime_and_gate_lanes_do_not_promote_ladder():
    lanes = {lane["id"]: lane for lane in _contract()["orthogonal_supporting_lanes"]}

    assert lanes["OL1"]["current_status"] == "supported_bounded"
    assert lanes["OL1"]["does_not_promote_to"] == ["CL2", "CL3", "CL4"]
    assert lanes["OL2"]["current_status"] == "gate_only_current_host_no_go"
    assert lanes["OL2"]["does_not_promote_to"] == ["CL3", "CL4"]
    for lane in lanes.values():
        for relative_path in lane["evidence_paths"]:
            assert (ROOT / relative_path).exists(), relative_path


def test_claim_registry_is_closed_world_and_blocked_claims_have_no_level():
    contract = _contract()
    claims = contract["claim_registry"]
    known_levels = set(contract["level_order"])
    known_lanes = {lane["id"] for lane in contract["orthogonal_supporting_lanes"]}

    assert [claim["id"] for claim in claims] == [f"PC{i:02d}" for i in range(1, 9)]
    for claim in claims:
        assert claim["allowed_wording"]
        assert claim["forbidden_wording"]
        if claim["paper_status"] == "blocked":
            assert claim["current_level"] is None
            assert claim["allowed_wording"].startswith("[Evidence needed:")
        elif claim.get("current_level") is not None:
            assert claim["current_level"] in known_levels
        else:
            assert claim["supporting_lane"] in known_lanes


def test_human_contract_covers_every_machine_readable_id_and_no_go_boundary():
    text = HUMAN_PATH.read_text(encoding="utf-8")
    contract = _contract()

    for identifier in contract["level_order"]:
        assert f"`{identifier}`" in text
    for claim in contract["claim_registry"]:
        assert f"`{claim['id']}`" in text
    assert "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE" in text
    assert "software-HIL 属于 CL1" in text
    assert "不是严格可互换的单标量" in text


def test_current_t24_anchor_still_matches_the_frozen_claim():
    summary_path = (
        ROOT
        / "runs"
        / "p4_benchmark"
        / "T24_formal_software_revalidation_20260510_200743"
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["missing_runs"] == []
    assert len(summary["comparison_rows"]) == 20
    assert len(summary["raw_rows"]) == 40
    assert all(row["completed_repeats"] == 2 for row in summary["comparison_rows"])
    assert all(row["expected_repeats"] == 2 for row in summary["comparison_rows"])
    assert all(row["coverage"] == 1.0 for row in summary["comparison_rows"])
    assert len(summary["scenario_winners"]) == 4
    assert all(
        row["best_mode"] == "hybrid_residual_b"
        for row in summary["scenario_winners"]
    )


def test_current_runtime_and_real_board_gate_verdicts_remain_separate():
    t48 = json.loads(
        (
            ROOT
            / "artifacts"
            / "t48_true_tflite_runtime_gate"
            / "t48_true_tflite_runtime_gate.json"
        ).read_text(encoding="utf-8")
    )
    t72 = json.loads(
        (
            ROOT
            / "artifacts"
            / "t72_real_board_transfer_pack_provenance_hardening"
            / "current_host_regenerated_gate.json"
        ).read_text(encoding="utf-8")
    )

    assert t48["final_gate_verdict"] == "GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8"
    assert (
        t72["final_gate_verdict"]
        == "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE"
    )
    assert t72["device_path_truth"]["status"] == "not_ready"
    assert t72["repo_execution_path_truth"]["status"] == "placeholder_only"
