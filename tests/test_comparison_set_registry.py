from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.comparison_set_registry import (
    COMPARISON_LANES,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    REQUIRED_COMPARATOR_IDS,
    build_comparison_set_registry,
    comparison_specs,
    validate_comparator_specs,
    write_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def test_registry_has_exact_required_set_and_pass_semantics(artifact: dict) -> None:
    assert artifact["task_id"] == "T5.1.1"
    assert artifact["status"] == "PASS"
    assert artifact["full_matrix_status"] == "PREREGISTERED_NOT_EXECUTED_T5_1_2"
    assert tuple(row["comparator_id"] for row in artifact["comparators"]) == REQUIRED_COMPARATOR_IDS
    assert len(artifact["comparators"]) == 19
    assert len(artifact["comparison_lanes"]) == 8
    assert artifact["gate_summary"] == {"passed": 14, "total": 14, "failed": []}
    assert "not a claim that one global" in artifact["pass_semantics"]


def test_all_comparators_have_full_scope_and_reciprocal_lane_membership(
    artifact: dict,
) -> None:
    by_id = {row["comparator_id"]: row for row in artifact["comparators"]}
    required = {
        "decision_target",
        "information_set",
        "output_contract",
        "protocol_scope",
        "metric_scope",
        "time_basis",
        "compute_budget",
        "deployability",
        "ranking_status",
        "eligible_lanes",
        "implementation_path",
        "claim_boundary",
    }
    for row in by_id.values():
        assert required <= row.keys()
        assert row["information_set"] and row["eligible_lanes"]
    for lane_id, contract in artifact["comparison_lanes"].items():
        assert contract["members"]
        assert contract["metric_contract"]
        assert contract["ranking_rule"]
        for member in contract["members"]:
            assert lane_id in by_id[member]["eligible_lanes"]
    assert set(artifact["comparison_lanes"]) == set(COMPARISON_LANES)


def test_only_decoder_oracle_reads_hidden_truth_and_oracles_are_not_deployable(
    artifact: dict,
) -> None:
    readers = [row for row in artifact["comparators"] if row["hidden_truth_access"]]
    assert [row["comparator_id"] for row in readers] == ["decoder_oracle_map"]
    assert readers[0]["deployability"] == "nondeployable"
    control = next(
        row
        for row in artifact["comparators"]
        if row["comparator_id"] == "finite_horizon_control_oracle"
    )
    assert control["deployability"] == "nondeployable"
    assert control["eligible_lanes"] == ["control_oracle_short_horizon"]
    assert "two-cycle" in control["claim_boundary"]


def test_no_correction_probe_is_zero_action_physical_idle_not_sbs_alias(
    artifact: dict,
) -> None:
    probe = artifact["no_correction_probe"]
    assert all(probe["gates"].values())
    assert probe["time_us"] == [0.0, 10.0, 20.0, 30.0]
    for field in (
        "measurement_events",
        "reset_events",
        "active_gate_applications",
        "frame_updates",
        "outcome_dependent_parameter_updates",
    ):
        assert probe["event_accounting"][field] == 0
    assert probe["standard_sbs_density_max_difference"] > 0.3
    assert probe["ten_us_vs_two_five_us_semigroup_error"] <= 2.0e-12
    assert probe["minimum_final_eigenvalue"] >= -2.0e-12


def test_finite_energy_static_probe_is_train_eval_separated_five_point_execution(
    artifact: dict,
) -> None:
    probe = artifact["finite_energy_static_probe"]
    assert probe["scope"] == "syndrome_level_effective_model"
    assert probe["train_samples"] == 120_000
    assert probe["eval_samples"] == 300_000
    assert len(probe["points"]) == 5
    assert all(probe["gates"].values())
    gains = [row["fitted_gain"] for row in probe["points"]]
    assert gains == sorted(gains)
    assert all(row["gain_ci_low"] > 0.0 for row in probe["points"])


def test_component_rows_and_secondary_protocols_cannot_enter_main_ranking(
    artifact: dict,
) -> None:
    by_id = {row["comparator_id"]: row for row in artifact["comparators"]}
    for comparator_id in ("run_length_event_controller", "regime_hmm_estimator"):
        assert by_id[comparator_id]["ranking_status"] == "component_only"
        assert by_id[comparator_id]["eligible_lanes"] == [
            "event_and_regime_components"
        ]
    exclusions = {row["protocol_id"]: row for row in artifact["secondary_exclusions"]}
    assert set(exclusions) == {"secondary_knill_qunaught", "secondary_psteane"}
    assert all("NOT_MAIN_RANKING" in row["status"] for row in exclusions.values())


def test_mf_student_and_teacher_claim_boundaries_preserve_counterevidence(
    artifact: dict,
) -> None:
    by_id = {row["comparator_id"]: row for row in artifact["comparators"]}
    assert "cutoff reversal" in by_id["latest_outcome_mf_fnn"]["claim_boundary"]
    assert "not global optimum" in by_id["bounded_residual_rnn_teacher"][
        "claim_boundary"
    ]
    assert "no universal NMF" in by_id["distilled_low_dimensional_student"][
        "claim_boundary"
    ]
    assert "87 analytic MACs" in by_id["distilled_low_dimensional_student"][
        "compute_budget"
    ]


def test_parent_artifacts_and_implementation_bindings_are_current(artifact: dict) -> None:
    assert len(artifact["artifact_bindings"]) == 16
    assert len(artifact["implementation_bindings"]) == 19
    for binding in artifact["artifact_bindings"]:
        path = ROOT / binding["path"]
        assert binding["machine_pass"] is True
        assert binding["status"] == "PASS"
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for binding in artifact["implementation_bindings"]:
        path = ROOT / binding["path"]
        assert binding["expected_fragment"] in path.read_text(encoding="utf-8")
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_data_has_complete_row_types_and_marks_no_global_matrix(
    artifact: dict,
) -> None:
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == artifact["source_data"]["row_count"] == 100
    counts = {
        row_type: sum(row["row_type"] == row_type for row in rows)
        for row_type in {row["row_type"] for row in rows}
    }
    assert counts == {
        "comparator": 19,
        "artifact": 16,
        "implementation": 19,
        "lane_member": 28,
        "exclusion": 2,
        "probe": 2,
        "gate": 14,
    }
    assert not any(row["family"] == "global_leaderboard" for row in rows)
    assert artifact["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()


def test_validation_fails_closed_on_missing_hidden_truth_and_lane_mutations() -> None:
    specs = comparison_specs()
    with pytest.raises(ValueError, match="frozen required order"):
        validate_comparator_specs(specs[:-1])
    mutated_truth = (replace(specs[0], hidden_truth_access=True), *specs[1:])
    with pytest.raises(ValueError, match="only decoder_oracle_map"):
        validate_comparator_specs(mutated_truth)
    standard_index = REQUIRED_COMPARATOR_IDS.index("standard_binning")
    mutated = list(specs)
    mutated[standard_index] = replace(
        mutated[standard_index],
        eligible_lanes=("decoder_continuous_drift",),
    )
    with pytest.raises(ValueError, match="reciprocally"):
        validate_comparator_specs(tuple(mutated))


def test_repeated_build_and_writer_preserve_contract_and_provenance(tmp_path: Path) -> None:
    first = build_comparison_set_registry()
    second = build_comparison_set_registry()
    assert first["contract_sha256"] == second["contract_sha256"]
    artifact_path = tmp_path / "comparison.json"
    source_path = tmp_path / "comparison.csv"
    written = write_artifacts(artifact_path, source_path)
    reloaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert written["status"] == reloaded["status"] == "PASS"
    assert reloaded["contract_sha256"] == first["contract_sha256"]
    assert reloaded["source_data"]["sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()


def test_human_report_keeps_lane_and_readiness_boundaries() -> None:
    report = (ROOT / "docs" / "comparison_set_registry.md").read_text(encoding="utf-8")
    for token in (
        "19 个 comparator",
        "不是一张全局排行榜",
        "PREREGISTERED_NOT_EXECUTED_T5_1_2",
        "No correction",
        "Knill/P-Steane 不进入 sBs 主排名",
        "universal NMF",
    ):
        assert token in report
