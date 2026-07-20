from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import secondary_experiment_preregistration as prereg


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_16_3_secondary_preregistration.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _experiments(report: dict) -> dict[str, dict]:
    return {row["task_id"]: row for row in report["experiments"]}


def test_report_recomputes_all_gates_and_preserves_readonly_boundary() -> None:
    report = _report()
    prereg.verify_report(report)
    assert report["verdict"] == "PASS_PHASE6C_READONLY_SECONDARY_PREREGISTRATION"
    assert report["gate_summary"] == {"passed": 17, "failed": []}
    assert all(report["live_phase6b_lock_checks"].values())
    assert len(report["live_input_lock_checks"]) == 14
    assert all(report["live_input_lock_checks"].values())
    contract = report["phase6b_lock"]["frozen_main_contract"]
    assert contract["v5_relative_ler_gate"] == 0.10
    assert contract["incremental_action_space_gate"] == 0.12
    assert contract["phase6c_may_change_phase6b"] is False
    assert contract["phase6c_may_rescue_v5"] is False
    assert contract["phase6c_may_unblock_t6_9_2"] is False


def test_nine_tasks_have_exact_schema_and_unique_secondary_seeds() -> None:
    report = _report()
    assert {row["task_id"] for row in report["experiments"]} == prereg.EXPECTED_TASKS
    assert all(set(row) == prereg.EXPERIMENT_FIELDS for row in report["experiments"])
    seeds = [
        seed
        for row in report["experiments"]
        if row["seeds"]["namespace"] == "phase6c-secondary-v1"
        for seed in row["seeds"]["values"]
    ]
    assert len(seeds) == len(set(seeds))
    assert all(61_710_001 <= seed <= 61_839_999 for seed in seeds)


def test_statistics_freeze_cluster_resampling_crn_and_no_result_stopping() -> None:
    report = _report()
    stats = report["statistics"]
    assert stats["confidence_level"] == 0.95
    assert stats["paired_bootstrap_resamples"] == 20_000
    assert stats["threshold_bootstrap_resamples"] == 2_000
    assert "seed cluster" in stats["resampling_unit"]
    assert "never pool" in stats["multiplicity"]
    assert "no performance-dependent" in stats["selection_rule"]
    for row in report["experiments"]:
        assert row["pairing"]
        assert row["runtime_budget"]["failure_on_exceed"] in row["failure_branches"]
        assert "favorable" not in row["stopping_rule"].lower() or "never" in row["stopping_rule"].lower()


def test_cnot_reproduction_has_fixed_anchor_sample_and_source_failure_branch() -> None:
    exp = _experiments(_report())["T6.17.2"]
    assert exp["config"]["squeezing_db"] == [9, 12, 13]
    assert exp["sample_size"] == {
        "target_failures_per_method_point": 2000,
        "n_max_per_squeezing_point": 5_000_000,
        "minimum_boundary_samples": 100_000,
    }
    assert len(exp["seeds"]["values"]) == 32
    assert exp["seeds"]["kind"] == "common_random_numbers"
    assert "identical primitive Gaussian draws" in exp["pairing"]
    assert "BLOCKED_SOURCE_INCOMPLETE" in exp["failure_branches"]
    assert any("tolerance" in action.lower() for action in exp["forbidden_actions"])


def test_checkpoint_eligibility_is_readonly_and_cannot_reselect_or_train() -> None:
    exp = _experiments(_report())["T6.17.3"]
    assert exp["execution_type"] == "READONLY_CHECKPOINT_REPLAY"
    assert exp["config"]["training_allowed"] is False
    assert exp["config"]["checkpoint_reselection_allowed"] is False
    assert exp["seeds"]["values"] == []
    assert any("select a better checkpoint" in action for action in exp["forbidden_actions"])


def test_aqec_protocol_has_common_wallclock_and_no_desired_ordering() -> None:
    exp = _experiments(_report())["T6.18.1"]
    assert exp["config"]["common_horizon_us"] == 700
    assert exp["config"]["cutoffs"] == [12, 16]
    assert len(exp["seeds"]["values"]) == 24
    assert "no desired ordering gate" in exp["stopping_rule"]
    assert exp["sources"] == ["LACHANCE2024_AQEC"]


def test_official_cpd_commit_and_conditional_multimode_scope_are_frozen() -> None:
    experiments = _experiments(_report())
    cpd = experiments["T6.18.2"]
    assert cpd["source_code"] == {
        "url": "https://github.com/amazon-science/LatticeAlgorithms.jl",
        "commit": "01f9bf1f6970b3e229b43aac9da3325c75518db8",
        "license": "Apache-2.0",
        "state": "PINNED_NOT_YET_IMPORTED",
    }
    assert cpd["config"]["surface_gkp_sizes"] == [3, 5, 7]
    assert cpd["config"]["sigma_grid"] == pytest.approx([0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64])
    extension = experiments["T6.18.3"]
    assert extension["entry_gate"].endswith("NOT_RUN_SCOPE_GATE")
    assert "NOT_RUN_SCOPE_GATE" in extension["failure_branches"]
    assert any("rescue task" in action for action in extension["forbidden_actions"])


def test_hardware_profile_keeps_absent_rtl_na_and_board_measurements_null() -> None:
    exp = _experiments(_report())["T6.19.1"]
    assert exp["config"]["board_measured_fields"] == "NULL_UNTIL_T6.9.2"
    assert "N_A_NO_RTL" in exp["failure_branches"]
    assert any("demo RTL" in action for action in exp["forbidden_actions"])
    assert exp["sample_size"]["integer_cxxrtl_action_mismatches_allowed"] == 0


def test_semantic_hash_ignores_only_metadata_not_scientific_state() -> None:
    v5 = prereg._load(prereg.T6155)
    baseline = prereg._canonical_sha256(prereg._v5_semantic(v5))
    metadata_only = deepcopy(v5)
    metadata_only["generated_at_utc"] = "2099-01-01T00:00:00+00:00"
    metadata_only["parent_bindings"] = {"regenerated": "metadata"}
    assert prereg._canonical_sha256(prereg._v5_semantic(metadata_only)) == baseline
    scientific_change = deepcopy(v5)
    scientific_change["headroom_recomputation"]["router_gate"] = 0.01
    assert prereg._canonical_sha256(prereg._v5_semantic(scientific_change)) != baseline


def test_source_csv_bindings_and_one_mutation_per_gate_are_complete() -> None:
    report = _report()
    csv_path = ROOT / report["source_data"]["path"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 158
    assert {row["record_type"] for row in rows} == {
        "experiment", "seed", "source_lock", "parent_lock", "phase6b_lock"
    }
    for binding in report["bindings"].values():
        path = ROOT / binding["path"]
        assert path.exists()
        assert prereg._sha256(path) == binding["sha256"]
    mutations = report["semantic_mutation_audit"]
    assert mutations["count"] == mutations["detected"] == 17
    assert {row["target_gate"] for row in mutations["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in mutations["cases"])


def test_forged_gate_upgrade_or_atlas_promotion_invalidates_report() -> None:
    report = _report()
    forged = deepcopy(report)
    forged["phase6b_lock"]["frozen_main_contract"]["v5_relative_ler_gate"] = 0.01
    with pytest.raises(ValueError, match="gates"):
        prereg.verify_report(forged)
    forged = deepcopy(report)
    _experiments(forged)["T6.19.3"]["config"]["may_upgrade_phase6b"] = True
    with pytest.raises(ValueError, match="gates"):
        prereg.verify_report(forged)


def test_markdown_is_valid_utf8_and_contains_no_replacement_characters() -> None:
    text = prereg.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert "二级实验预注册与只读边界" in text
    assert "当前工具状态不是结果" in text
    assert "\ufffd" not in text
    assert "158 rows" in text
