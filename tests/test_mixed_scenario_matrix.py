from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from math import isclose
from pathlib import Path

import pytest

from cnn_fpga.benchmark.mixed_scenario_matrix import (
    DECODER_METHODS,
    DECODER_SCENARIO_IDS,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    REQUIRED_SCENARIO_IDS,
    MixedDecoderScenario,
    validate_matrix_payload,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ROOT / DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


def test_artifact_pass_means_lane_local_execution_not_global_superiority(
    artifact: dict,
) -> None:
    assert artifact["task_id"] == "T5.1.2"
    assert artifact["status"] == "PASS"
    assert artifact["matrix_status"] == "EXECUTED_LANE_LOCAL_NO_CROSS_LANE_RANKING"
    assert artifact["gate_summary"] == {"passed": 15, "total": 15, "failed": []}
    assert "no universal comparator superiority" in artifact["pass_semantics"]
    assert "global leaderboard" in artifact["claim_boundary"]["forbidden"]
    assert "global_leaderboard" not in artifact


def test_exact_ten_scenarios_have_one_explicit_execution_lane(artifact: dict) -> None:
    assert tuple(artifact["required_scenarios"]) == REQUIRED_SCENARIO_IDS
    assert set(artifact["scenario_execution_map"]) == set(REQUIRED_SCENARIO_IDS)
    assert tuple(artifact["decoder_lane"]["executed_scenarios"]) == DECODER_SCENARIO_IDS
    assert artifact["loss_lane"]["scenario_id"] == "loss"
    assert artifact["readout_ancilla_lane"]["scenario_id"] == "readout_ancilla_drift"
    assert artifact["large_error_lane"]["scenario_id"] == "large_error_recovery"
    assert artifact["leakage_lane"]["scenario_id"] == "leakage"


def test_decoder_lane_uses_disjoint_seeds_unique_shared_traces_and_fixed_methods(
    artifact: dict,
) -> None:
    lane = artifact["decoder_lane"]
    config = lane["config"]
    assert set(config["training_seeds"]).isdisjoint(config["evaluation_seeds"])
    assert tuple(lane["executed_comparators"]) == DECODER_METHODS
    assert "only after current-window decoding" in lane["shared_trace_contract"]
    assert "before all T5.1.2 evaluation" in lane["training_contract"]
    assert len(lane["seed_rows"]) == 36
    assert len({row["trace_sha256"] for row in lane["seed_rows"]}) == 36
    for scenario_id in DECODER_SCENARIO_IDS:
        rows = [row for row in lane["seed_rows"] if row["scenario_id"] == scenario_id]
        assert len(rows) == 6
        assert all(row["evaluation_samples"] == 32 * 512 for row in rows)
        aggregate = next(
            row for row in lane["scenario_aggregates"] if row["scenario_id"] == scenario_id
        )
        assert aggregate["evaluation_samples"] == 6 * 32 * 512
        assert aggregate["unique_trace_hashes"] == 6
        assert set(aggregate["methods"]) == set(DECODER_METHODS)


def test_decoder_scenario_generators_are_nontrivial_and_separate_failure_modes() -> None:
    windows = 32
    static = MixedDecoderScenario("static_gaussian").states(windows)
    assert len({(s.mu_q, s.mu_p, s.sigma_q, s.sigma_p, s.rho) for s in static}) == 1

    mean = MixedDecoderScenario("mean_drift").states(windows)
    assert mean[0].mu_q < mean[-1].mu_q and mean[0].mu_p > mean[-1].mu_p
    assert len({(s.sigma_q, s.sigma_p, s.rho) for s in mean}) == 1

    variance = MixedDecoderScenario("variance_drift").states(windows)
    assert variance[0].sigma_q < variance[-1].sigma_q
    assert variance[0].sigma_p > variance[-1].sigma_p
    assert len({s.rho for s in variance}) == 1

    correlation = MixedDecoderScenario("correlation_drift").states(windows)
    assert correlation[0].rho < -0.7 and correlation[-1].rho > 0.7
    assert len({(s.sigma_q, s.sigma_p) for s in correlation}) == 1

    burst = MixedDecoderScenario("burst_outlier").states(windows)
    assert 5 <= sum(s.burst_active for s in burst) < windows / 2
    assert max(s.p_outlier for s in burst) == pytest.approx(0.10)
    assert max(s.outlier_scale for s in burst) == pytest.approx(4.5)
    assert {s.event_id for s in burst} == {0, 1, 2}

    shift = MixedDecoderScenario("calibration_shift").states(windows)
    assert all(s.event_id == 0 for s in shift[: windows // 2])
    assert all(s.event_id == 1 for s in shift[windows // 2 :])
    assert shift[windows // 2 - 1].mean.tolist() != shift[windows // 2].mean.tolist()
    with pytest.raises(ValueError, match="unknown mixed decoder"):
        MixedDecoderScenario("loss")


def test_decoder_results_report_signed_counterevidence_without_success_gates(
    artifact: dict,
) -> None:
    lane = artifact["decoder_lane"]
    assert lane["frozen_hyperparameters"]["ewma_alpha"] == pytest.approx(0.85)
    for scenario in lane["scenario_aggregates"]:
        for method in DECODER_METHODS:
            estimate = scenario["methods"][method]["error_rate_seed_cluster_ci"]["estimate"]
            assert 0.0 <= estimate <= 1.0
        assert scenario["paired_fixed_contrasts"]["static_minus_oracle"]["estimate"] >= 0.0
    assert not any(
        token in gate
        for gate in artifact["gates"]
        for token in ("improves", "superior", "wins", "best")
    )
    assert "different decision target" in lane["nonexecution_reason"]


def test_loss_lane_is_an_isolated_monotone_physics_sweep(artifact: dict) -> None:
    lane = artifact["loss_lane"]
    rows = lane["rows"]
    assert all(lane["gates"].values())
    assert [row["loss_transmissivity"] for row in rows] == [1.0, 0.98, 0.94, 0.88]
    assert rows[0]["loss_bias_norm"] == pytest.approx(0.0, abs=1.0e-15)
    assert all(
        right["loss_bias_norm"] > left["loss_bias_norm"]
        for left, right in zip(rows, rows[1:])
    )
    assert all(
        right["decision_covariance_trace"] > left["decision_covariance_trace"]
        for left, right in zip(rows, rows[1:])
    )
    for row in rows:
        expected = 1.0 - (1.0 - row["q_odd_alias_probability"]) * (
            1.0 - row["p_odd_alias_probability"]
        )
        assert isclose(row["any_jump_probability"], expected, rel_tol=0.0, abs_tol=2e-15)
    assert rows[0]["validity"] == "localized"
    assert rows[-1]["validity"] == "clipping_dominated"


def test_readout_ancilla_lane_has_rate_sweep_and_protocol_native_endpoint(
    artifact: dict,
) -> None:
    lane = artifact["readout_ancilla_lane"]
    assert all(lane["gates"].values())
    assert len(lane["rows"]) == 4
    for row in lane["rows"]:
        for key in (
            "big_cd_bit_z_score",
            "readout_mismatch_z_score",
            "logical_backaction_z_score",
        ):
            assert abs(row[key]) <= 5.0
    assert all(lane["production_endpoint"]["checks"].values())
    assert len(lane["production_endpoint"]["secondary_protocols"]) == 3
    assert all(
        not protocol["executable"]
        for protocol in lane["production_endpoint"]["secondary_protocols"]
    )


def test_large_error_and_leakage_keep_native_component_gates(artifact: dict) -> None:
    large = artifact["large_error_lane"]
    leakage = artifact["leakage_lane"]
    assert large["ranking_status"] == leakage["ranking_status"] == (
        "component_only_not_decoder_leaderboard"
    )
    assert large["result"]["gate"]["passed"] is True
    assert len(large["result"]["points"]) == 9
    assert all(check["passed"] for check in large["result"]["gate"]["checks"])
    assert leakage["result"]["gate"]["passed"] is True
    assert all(check["passed"] for check in leakage["result"]["gate"]["checks"])
    assert leakage["result"]["tail_correlation"]["retained_fraction"] < 0.95


def test_artifact_implementation_and_source_data_provenance_are_current(
    artifact: dict,
) -> None:
    for binding in artifact["artifact_bindings"] + artifact["implementation_bindings"]:
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    source_path = ROOT / artifact["source_data"]["path"]
    assert source_path == ROOT / DEFAULT_SOURCE_DATA
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == artifact["source_data"]["sha256"]
    with source_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == artifact["source_data"]["row_count"] == 116
    counts = {
        kind: sum(row["row_type"] == kind for row in rows)
        for kind in {row["row_type"] for row in rows}
    }
    assert counts == {
        "scenario": 10,
        "decoder_seed": 36,
        "decoder_method": 36,
        "loss": 4,
        "ancilla_drift": 4,
        "ancilla_endpoint": 1,
        "large_error": 9,
        "leakage": 1,
        "gate": 15,
    }
    assert not any(row["lane_id"] == "global_leaderboard" for row in rows)


def test_validator_fails_closed_on_pairing_provenance_and_nonmixing_mutations(
    artifact: dict,
) -> None:
    assert len(validate_matrix_payload(artifact)) == 5

    missing = deepcopy(artifact)
    missing["required_scenarios"].pop()
    with pytest.raises(ValueError, match="required scenario"):
        validate_matrix_payload(missing)

    overlap = deepcopy(artifact)
    overlap["decoder_lane"]["config"]["evaluation_seeds"][0] = overlap["decoder_lane"]["config"]["training_seeds"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_matrix_payload(overlap)

    mixed = deepcopy(artifact)
    mixed["large_error_lane"]["ranking_status"] = "main_decoder_ranking"
    with pytest.raises(ValueError, match="component-only"):
        validate_matrix_payload(mixed)

    stale = deepcopy(artifact)
    stale["artifact_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale or failed"):
        validate_matrix_payload(stale)

    leaderboard = deepcopy(artifact)
    leaderboard["global_leaderboard"] = []
    with pytest.raises(ValueError, match="forbidden"):
        validate_matrix_payload(leaderboard)

    failed = deepcopy(artifact)
    failed["gates"]["loss_native_gates_pass"] = False
    with pytest.raises(ValueError, match="fifteen"):
        validate_matrix_payload(failed)


def test_human_report_preserves_execution_and_claim_boundaries() -> None:
    report = (ROOT / "docs" / "mixed_scenario_matrix.md").read_text(encoding="utf-8")
    for token in (
        "10 类场景",
        "不是全局排行榜",
        "36 个 seed-cluster",
        "burst/outlier",
        "Fréchet",
        "component-only",
        "不代表算法优势",
    ):
        assert token in report


def test_protocol_hierarchy_registers_lane_local_matrix_contract() -> None:
    hierarchy = json.loads(
        (ROOT / "docs" / "protocol_hierarchy.json").read_text(encoding="utf-8")
    )
    contract = hierarchy["mixed_scenario_matrix_contract"]
    assert contract["task_id"] == "T5.1.2"
    assert contract["scenario_count"] == 10
    assert contract["decoder_seed_clusters"] == 36
    assert contract["decoder_paired_decisions"] == 589_824
    assert contract["matrix_status"] == "EXECUTED_LANE_LOCAL_NO_CROSS_LANE_RANKING"
    assert "never enter" in contract["component_rule"]
