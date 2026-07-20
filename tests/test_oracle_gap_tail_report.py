from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.oracle_gap_tail_report import (
    BOOTSTRAP_REPLICATES,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    MULTIPLICITY_METHOD,
    _exact_sign_flip_pvalue,
    _holm_adjust,
    validate_report_payload,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ROOT / DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


def test_pass_is_reporting_complete_not_algorithm_success(artifact: dict) -> None:
    assert artifact["task_id"] == "T5.1.3"
    assert artifact["status"] == "PASS"
    assert artifact["gate_summary"] == {"passed": 15, "total": 15, "failed": []}
    assert "not the T5.1.4 algorithm-success verdict" in artifact["pass_semantics"]
    assert "T5.1.4 success claim" in artifact["claim_boundary"]["forbidden"]


def test_window_replay_exactly_reproduces_t5_1_2_seed_traces_and_rates(
    artifact: dict,
) -> None:
    lane = artifact["decoder_lane"]
    assert len(lane["seed_rows"]) == 36
    assert len(lane["window_rows"]) == 1152
    assert all(row["t5_1_2_trace_match"] for row in lane["seed_rows"])
    assert max(row["t5_1_2_max_error_rate_difference"] for row in lane["seed_rows"]) <= 1e-15
    assert len({row["trace_sha256"] for row in lane["seed_rows"]}) == 36
    assert all(row["samples"] == 512 for row in lane["window_rows"])
    assert {
        (row["scenario_id"], row["evaluation_seed"], row["window_id"])
        for row in lane["window_rows"]
    }.__len__() == 1152


def test_average_p95_and_worst_use_seed_cluster_bootstrap(artifact: dict) -> None:
    lane = artifact["decoder_lane"]
    assert lane["independent_unit"] == "evaluation_seed"
    assert lane["bootstrap"]["replicates"] == BOOTSTRAP_REPLICATES == 20_000
    assert "resample six whole seed trajectories" in lane["bootstrap"]["p95_rule"]
    assert "do not present a naive iid-window CI" in lane["bootstrap"]["worst_rule"]
    for scenario in lane["scenario_reports"]:
        assert scenario["window_count"] == 192
        assert len(scenario["seed_order"]) == 6
        for metrics in scenario["methods"].values():
            assert 0.0 <= metrics["p_l"] <= metrics["observed_worst_window_ler"] <= 1.0
            assert metrics["p_l"] <= metrics["window_ler_p95"] <= metrics["observed_worst_window_ler"]
            for key in (
                "p_l_bootstrap_ci",
                "window_ler_p95_bootstrap_ci",
                "mean_per_seed_worst_bootstrap_ci",
            ):
                assert metrics[key]["replicates"] == 20_000
                assert metrics[key]["method"] == "paired_seed_cluster_percentile_bootstrap"


def test_tail_report_preserves_calibration_shift_transient_counterevidence(
    artifact: dict,
) -> None:
    scenario = next(
        row
        for row in artifact["decoder_lane"]["scenario_reports"]
        if row["scenario_id"] == "calibration_shift"
    )
    static = scenario["methods"]["static"]
    kalman = scenario["methods"]["kalman"]
    assert kalman["p_l"] < static["p_l"]
    assert kalman["window_ler_p95"] < static["window_ler_p95"]
    assert kalman["observed_worst_window_ler"] > static["observed_worst_window_ler"]
    assert kalman["observed_worst_window_ler"] == pytest.approx(55 / 512)
    assert static["observed_worst_window_ler"] == pytest.approx(37 / 512)


def test_decoder_oracle_gaps_are_paired_signed_and_denominator_reliable(
    artifact: dict,
) -> None:
    for scenario in artifact["decoder_lane"]["scenario_reports"]:
        static = scenario["methods"]["static"]["decoder_oracle_gap"]
        oracle = scenario["methods"]["oracle"]["decoder_oracle_gap"]
        assert static["gap_closed_fraction"] == pytest.approx(0.0, abs=1e-15)
        assert oracle["gap_closed_fraction"] == pytest.approx(1.0, abs=1e-15)
        for method in scenario["methods"].values():
            gap = method["decoder_oracle_gap"]
            assert gap["static_oracle_gap"] > 0.0
            assert gap["bootstrap_valid_fraction"] == 1.0
            assert gap["bootstrap_valid_replicates"] == 20_000
            assert gap["bootstrap_total_replicates"] == 20_000
            assert "positive" in gap["denominator_rule"]


def test_multiplicity_family_is_exact_seed_level_holm_and_has_zero_discoveries(
    artifact: dict,
) -> None:
    multiplicity = artifact["decoder_lane"]["multiplicity"]
    assert multiplicity["hypotheses"] == 24
    assert multiplicity["adjustment"] == MULTIPLICITY_METHOD
    assert multiplicity["discoveries"] == 0
    assert min(row["raw_p_value"] for row in multiplicity["rows"]) == pytest.approx(0.03125)
    assert min(row["holm_adjusted_p_value"] for row in multiplicity["rows"]) == pytest.approx(0.75)
    assert all(row["paired_seed_count"] == 6 for row in multiplicity["rows"])
    assert all(not row["reject_at_familywise_alpha_0_05"] for row in multiplicity["rows"])

    assert _exact_sign_flip_pvalue([1, 2, 3, 4, 5, 6]) == pytest.approx(0.03125)
    rows = [
        {"raw_p_value": 0.01},
        {"raw_p_value": 0.04},
        {"raw_p_value": 0.03},
    ]
    adjusted = _holm_adjust(rows)
    assert [row["holm_adjusted_p_value"] for row in adjusted] == pytest.approx(
        [0.03, 0.06, 0.06]
    )


def test_control_oracle_gap_is_exact_two_cycle_and_preserves_negative_gaps(
    artifact: dict,
) -> None:
    lane = artifact["control_oracle_lane"]
    assert lane["uncertainty_status"] == "EXACT_BRANCH_EXPECTATION_NO_SAMPLING_CI"
    assert "false sampling interval" in lane["why_no_bootstrap_ci"]
    assert "not a globally certified" in lane["optimization_boundary"]
    assert [row["cutoff"] for row in lane["cutoffs"]] == [12, 16]
    for cutoff in lane["cutoffs"]:
        assert cutoff["full_cycles"] == 2
        assert cutoff["terminal_branches"] == 16
        assert cutoff["trajectory_probability_sum"] == pytest.approx(1.0, abs=1e-12)
        assert len(cutoff["methods"]) == 10
        oracle = next(
            row
            for row in cutoff["methods"]
            if row["method_id"] == "finite_horizon_control_oracle"
        )
        assert all(record["control_oracle_minus_method_gap"] == pytest.approx(0.0) for record in oracle["metrics"].values())
    cutoff12 = lane["cutoffs"][0]
    handcrafted = next(row for row in cutoff12["methods"] if row["method_id"] == "handcrafted_recurrence")
    assert handcrafted["metrics"]["terminal_fidelity"]["control_oracle_minus_method_gap"] > 0.0
    assert handcrafted["metrics"]["selection_score"]["control_oracle_minus_method_gap"] < 0.0
    cutoff16 = lane["cutoffs"][1]
    handcrafted16 = next(row for row in cutoff16["methods"] if row["method_id"] == "handcrafted_recurrence")
    assert handcrafted16["metrics"]["terminal_fidelity"]["control_oracle_minus_method_gap"] < 0.0


def test_source_data_and_all_bindings_are_current(artifact: dict) -> None:
    for binding in artifact["artifact_bindings"] + artifact["implementation_bindings"]:
        assert hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
    source_path = ROOT / artifact["source_data"]["path"]
    assert source_path == ROOT / DEFAULT_SOURCE_DATA
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == artifact["source_data"]["sha256"]
    with source_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == artifact["source_data"]["row_count"] == 7139
    counts = {
        row_type: sum(row["row_type"] == row_type for row in rows)
        for row_type in {row["row_type"] for row in rows}
    }
    assert counts == {
        "decoder_window": 6912,
        "decoder_summary": 108,
        "multiplicity": 24,
        "control_oracle_gap": 80,
        "gate": 15,
    }


def test_validator_fails_closed_on_tail_gap_multiplicity_and_horizon_mutations(
    artifact: dict,
) -> None:
    assert len(validate_report_payload(artifact)) == 6

    missing = deepcopy(artifact)
    missing["decoder_lane"]["window_rows"].pop()
    with pytest.raises(ValueError, match="1152"):
        validate_report_payload(missing)

    trace = deepcopy(artifact)
    trace["decoder_lane"]["seed_rows"][0]["t5_1_2_trace_match"] = False
    with pytest.raises(ValueError, match="trace mismatch"):
        validate_report_payload(trace)

    unreliable = deepcopy(artifact)
    first = unreliable["decoder_lane"]["scenario_reports"][0]["methods"]["kalman"]
    first["decoder_oracle_gap"]["bootstrap_valid_fraction"] = 0.5
    with pytest.raises(ValueError, match="not bootstrap reliable"):
        validate_report_payload(unreliable)

    multiplicity = deepcopy(artifact)
    multiplicity["decoder_lane"]["multiplicity"]["adjustment"] = "none"
    with pytest.raises(ValueError, match="adjustment drifted"):
        validate_report_payload(multiplicity)

    horizon = deepcopy(artifact)
    horizon["control_oracle_lane"]["cutoffs"][0]["full_cycles"] = 10
    with pytest.raises(ValueError, match="two-cycle"):
        validate_report_payload(horizon)

    stale = deepcopy(artifact)
    stale["artifact_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale or failed"):
        validate_report_payload(stale)

    failed = deepcopy(artifact)
    failed["gates"]["control_oracle_no_fake_sampling_ci"] = False
    with pytest.raises(ValueError, match="fifteen"):
        validate_report_payload(failed)


def test_human_report_keeps_tail_and_dual_oracle_counterevidence() -> None:
    report = (ROOT / "docs" / "oracle_gap_tail_report.md").read_text(encoding="utf-8")
    for token in (
        "1,152",
        "20,000",
        "Holm",
        "0 个",
        "55/512",
        "exact two-cycle",
        "不伪造 bootstrap CI",
        "不是 T5.1.4",
    ):
        assert token in report


def test_protocol_hierarchy_registers_tail_and_dual_oracle_contract() -> None:
    hierarchy = json.loads(
        (ROOT / "docs" / "protocol_hierarchy.json").read_text(encoding="utf-8")
    )
    contract = hierarchy["oracle_gap_tail_report_contract"]
    assert contract["task_id"] == "T5.1.3"
    assert contract["decoder_window_rows"] == 1152
    assert contract["bootstrap_replicates"] == 20_000
    assert contract["multiplicity_hypotheses"] == 24
    assert contract["multiplicity_discoveries"] == 0
    assert "not a globally certified" in contract["control_oracle_rule"]
