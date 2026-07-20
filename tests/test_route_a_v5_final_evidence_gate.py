from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from cnn_fpga.benchmark import route_a_v5_final_evidence_gate as gate


def _parents():
    headroom = json.loads(gate.HEADROOM_PATH.read_text(encoding="utf-8"))
    statuses = gate._task_statuses(gate.BOARD_PATH.read_text(encoding="utf-8"))
    board_gate = json.loads(gate.BOARD_GATE_PATH.read_text(encoding="utf-8"))
    v4_final = json.loads(gate.V4_FINAL_PATH.read_text(encoding="utf-8"))
    return headroom, statuses, board_gate, v4_final, gate._claim_rows()


def test_core_gate_recomputes_incremental_not_overall_oracle() -> None:
    parents = _parents()
    checks = gate._core_checks(*parents[:2], [], *parents[2:])
    assert all(checks.values())
    headroom = deepcopy(parents[0])
    expanded = headroom["development_audit"]["nested_audit"]["expanded_candidate_action_oracle"]
    expanded["incremental_action_space_headroom_vs_baseline"] = expanded["overall_relative_headroom_vs_baseline"]
    checks = gate._core_checks(headroom, parents[1], [], *parents[2:])
    assert checks["incremental_action_headroom_recomputes_below_gate"] is False


def test_all_conditional_tasks_are_dropped_and_outputs_absent() -> None:
    _, statuses, _, _, _ = _parents()
    assert len(gate.DOWNSTREAM_DROPPED_TASKS) == 20
    assert all(statuses[task] == "Dropped" for task in gate.DOWNSTREAM_DROPPED_TASKS)
    assert gate._v5_outputs() == []


def test_semantic_mutations_are_material_and_detected() -> None:
    headroom, statuses, board_gate, v4_final, claims = _parents()
    rows = gate._semantic_mutations(headroom, statuses, board_gate, v4_final, claims)
    assert len(rows) == 6
    assert all(row["detected"] for row in rows)
    assert all(row["failed_checks"] for row in rows)


def test_build_and_validate_report(tmp_path: Path) -> None:
    artifact = tmp_path / "report.json"
    source = tmp_path / "source.csv"
    report = gate.build_report(artifact, source)
    assert report["verdict"] == gate.VERDICT
    assert report["gate_summary"] == {"passed": 12, "failed": []}
    assert report["phase6c_permission"]["mode"] == "READ_ONLY_AUXILIARY_COMPARISONS"
    # Temporary artifacts cannot pass the repository-relative validator, but
    # all scientific gates and the independent analysis hash must still close.
    assert all(report["gates"].values())
    assert len(report["analysis_sha256"]) == 64


def test_repository_report_validates_when_present() -> None:
    if gate.DEFAULT_ARTIFACT.is_file():
        assert all(gate.validate_report().values())
