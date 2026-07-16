from __future__ import annotations

import csv
import hashlib
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.independent_cross_fidelity_holdout import (
    CALIBRATION_DB,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    EFFECTIVE_SAMPLES_PER_SEED,
    EXCLUDED_EXPLORATORY_PILOT_DB,
    HIGH_HOLDOUT_DB,
    HOLDOUT_SEEDS,
    MAIN_THRESHOLDS,
    NEGATIVE_HOLDOUT_DB,
    PSTEANE_B_GRID,
    PSTEANE_M_GRID,
    PSTEANE_SIGMA_A,
    PSTEANE_VARIANCE_RATIOS,
    build_holdout_report,
    implementation_sha256,
    psteane_covariance_propagation,
    psteane_small_noise_variances,
    psteane_variance_product_formula,
    write_artifacts,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def _points(payload: dict) -> dict[float, dict]:
    return {
        float(row["squeezing_db"]): row
        for row in payload["main_cross_fidelity_holdout"]["points"]
    }


def test_committed_artifact_is_current_and_source_bound() -> None:
    payload = _artifact()
    assert payload["task_id"] == "T5.0.2"
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["gate_summary"] == {"passed": 6, "total": 6, "failed": []}
    assert all(payload["acceptance_gates"].values())
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()


def test_formal_points_and_seeds_are_disjoint_from_calibration_and_pilot() -> None:
    payload = _artifact()
    contract = payload["preregistered_contract"]
    assert tuple(contract["calibration_db"]) == CALIBRATION_DB
    assert tuple(contract["excluded_exploratory_pilot"]["db"]) == EXCLUDED_EXPLORATORY_PILOT_DB
    assert contract["formal_negative_db"] == NEGATIVE_HOLDOUT_DB == 2.5
    assert tuple(contract["formal_high_db"]) == HIGH_HOLDOUT_DB == (10.25, 11.75)
    assert tuple(contract["seeds"]) == HOLDOUT_SEEDS
    assert contract["effective_samples_per_seed"] == EFFECTIVE_SAMPLES_PER_SEED
    assert not set([NEGATIVE_HOLDOUT_DB, *HIGH_HOLDOUT_DB]) & set(
        [*CALIBRATION_DB, *EXCLUDED_EXPLORATORY_PILOT_DB]
    )


def test_main_family_failure_is_preserved_and_does_not_fail_task_acceptance() -> None:
    payload = _artifact()
    main = payload["main_cross_fidelity_holdout"]
    assert main["status"] == "FAIL"
    assert main["failed_gates"] == [
        "pooled_effective_holdout_matches_noise_transfer_within_two_sigma"
    ]
    assert payload["secondary_psteane_holdout"]["status"] == "PASS"
    assert payload["acceptance_gates"]["at_least_one_independent_holdout_family_passes"] is True
    assert "hiding a failed family" in payload["claim_boundary"]["forbidden"]


def test_high_deterministic_lanes_pass_at_both_disjoint_points() -> None:
    points = _points(_artifact())
    for db in HIGH_HOLDOUT_DB:
        row = points[db]
        lanes = row["deterministic_lanes"]
        assert lanes["noise_vs_syndrome_q_ler_gap"] <= MAIN_THRESHOLDS[
            "noise_vs_syndrome_q_ler_gap_max"
        ]
        assert lanes["fock_vs_syndrome_q_ler_gap"] <= MAIN_THRESHOLDS[
            "fock_vs_syndrome_q_ler_gap_max"
        ]
        assert lanes["canonical_qp_ler_gap"] <= MAIN_THRESHOLDS[
            "canonical_qp_ler_gap_max"
        ]
        assert lanes["noise_transfer_validity"] == "localized"
        assert lanes["minimum_clipping_ratio"] >= 0.90
        assert row["effective_holdout"]["total_samples"] == 400_000


def test_effective_holdout_exposes_one_failed_and_one_passed_point() -> None:
    points = _points(_artifact())
    assert points[10.25]["effective_holdout"]["maximum_axis_z_score"] == pytest.approx(
        2.293337837390298
    )
    assert points[10.25]["effective_holdout"]["maximum_axis_z_score"] > 2.0
    assert points[11.75]["effective_holdout"]["maximum_axis_z_score"] == pytest.approx(
        1.8677649438942685
    )
    assert points[11.75]["effective_holdout"]["maximum_axis_z_score"] <= 2.0
    assert len(points[10.25]["effective_holdout"]["seed_rows"]) == 4


def test_negative_holdout_preserves_out_of_domain_failure() -> None:
    row = _points(_artifact())[NEGATIVE_HOLDOUT_DB]
    lanes = row["deterministic_lanes"]
    assert lanes["noise_vs_syndrome_q_ler_gap"] == pytest.approx(
        0.024242918915546363
    )
    assert lanes["minimum_clipping_ratio"] == pytest.approx(0.3256837629352086)
    assert lanes["noise_transfer_validity"] == "clipping_dominated"
    assert lanes["canonical_qp_ler_gap"] > 0.04


@pytest.mark.parametrize(
    "sigma_a,variance_ratio,b,m",
    [
        (0.07, 1.25, 0.5, 1),
        (0.11, 2.25, sqrt(2.0), 2),
        (0.19, 4.75, 2.5, 4),
    ],
)
def test_equation40_matches_independent_covariance_propagation(
    sigma_a: float, variance_ratio: float, b: float, m: int
) -> None:
    sigma_d = sqrt(variance_ratio) * sigma_a
    formula = psteane_small_noise_variances(sigma_d, sigma_a, b, m)
    independent = psteane_covariance_propagation(sigma_d, sigma_a, b, m)
    assert formula == pytest.approx(independent, rel=1e-13, abs=1e-15)
    assert formula[0] * formula[1] == pytest.approx(
        psteane_variance_product_formula(sigma_d, sigma_a, b, m),
        rel=1e-13,
        abs=1e-15,
    )


def test_psteane_m1_product_and_special_cases_are_exact() -> None:
    for sigma_a in PSTEANE_SIGMA_A:
        for ratio in PSTEANE_VARIANCE_RATIOS:
            sigma_d = sqrt(ratio) * sigma_a
            for b in PSTEANE_B_GRID:
                q, p = psteane_small_noise_variances(sigma_d, sigma_a, b, 1)
                assert q * p == pytest.approx(sigma_a**4, rel=1e-13, abs=1e-15)
            symmetric = psteane_small_noise_variances(
                sigma_d, sigma_a, sqrt(2.0), 1
            )
            assert symmetric == pytest.approx((sigma_a**2, sigma_a**2), abs=1e-15)


def test_psteane_m1_is_unique_grid_argmin_for_every_new_noise_ratio() -> None:
    for sigma_a in PSTEANE_SIGMA_A:
        for ratio in PSTEANE_VARIANCE_RATIOS:
            sigma_d = sqrt(ratio) * sigma_a
            for b in PSTEANE_B_GRID:
                products = {
                    m: np.prod(psteane_small_noise_variances(sigma_d, sigma_a, b, m))
                    for m in PSTEANE_M_GRID
                }
                assert min(products, key=products.get) == 1
                assert all(products[1] < products[m] for m in PSTEANE_M_GRID if m != 1)


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 0.1, 1.0, 1),
        (0.2, -0.1, 1.0, 1),
        (0.2, 0.1, 0.0, 1),
        (0.2, 0.1, 1.0, 0),
        (0.2, 0.1, 1.0, 1.5),
    ],
)
def test_psteane_formulas_reject_out_of_contract_inputs(args) -> None:
    with pytest.raises(ValueError):
        psteane_small_noise_variances(*args)
    with pytest.raises(ValueError):
        psteane_covariance_propagation(*args)
    with pytest.raises(ValueError):
        psteane_variance_product_formula(*args)


def test_secondary_artifact_covers_full_252_point_grid_and_all_gates() -> None:
    secondary = _artifact()["secondary_psteane_holdout"]
    assert secondary["grid"]["row_count"] == 252
    assert len(secondary["rows"]) == 252
    assert secondary["source"]["equations"] == [36, 37, 40, 41, 43]
    assert all(secondary["gates"].values())
    assert secondary["diagnostics"]["maximum_covariance_error"] < 1e-16
    assert secondary["diagnostics"]["maximum_product_relative_error"] < 1e-15
    assert secondary["diagnostics"]["argmin_failures"] == []
    assert "no sBs ranking" in secondary["scope"]


def test_parent_artifacts_remain_current_and_t501_snapshot_is_not_rewritten() -> None:
    payload = _artifact()
    for binding in payload["parent_artifacts"].values():
        path = Path(binding["path"])
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert binding["machine_pass"] is True
    registry = json.loads(
        Path("docs/t5_0_1_literature_trend_reproduction.json").read_text(
            encoding="utf-8"
        )
    )
    psteane = next(
        row for row in registry["targets"] if row["target_id"] == "LT-2026-PSTEANE-CONDITION"
    )
    assert psteane["current_status"] == "REGISTERED_PENDING"


def test_source_data_is_complete_and_marks_the_10_25_point_failed() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"] == 291
    counts = {
        row_type: sum(row["row_type"] == row_type for row in rows)
        for row_type in {row["row_type"] for row in rows}
    }
    assert counts == {
        "parent": 2,
        "main_point": 3,
        "effective_seed": 12,
        "psteane_grid": 252,
        "gate": 22,
    }
    point_rows = {row["record_id"]: row for row in rows if row["row_type"] == "main_point"}
    assert point_rows["db_10.25"]["passed"] == "False"
    assert point_rows["db_11.75"]["passed"] == "True"
    failed_gate = next(
        row
        for row in rows
        if row["record_id"]
        == "pooled_effective_holdout_matches_noise_transfer_within_two_sigma"
    )
    assert failed_gate["passed"] == "False"


def test_writer_round_trip_is_deterministic_and_preserves_failure(tmp_path: Path) -> None:
    artifact_path = tmp_path / "holdout.json"
    source_path = tmp_path / "holdout.csv"
    payload = write_artifacts(artifact_path, source_path)
    reloaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["contract_sha256"] == reloaded["contract_sha256"]
    assert reloaded["status"] == "PASS"
    assert reloaded["main_cross_fidelity_holdout"]["status"] == "FAIL"
    assert reloaded["secondary_psteane_holdout"]["status"] == "PASS"
    assert reloaded["source_data"]["sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()


def test_repeated_build_has_stable_contract_hash() -> None:
    first = build_holdout_report()
    second = build_holdout_report()
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["main_cross_fidelity_holdout"]["failed_gates"] == second[
        "main_cross_fidelity_holdout"
    ]["failed_gates"]


def test_human_report_preserves_main_failure_and_secondary_scope() -> None:
    report = Path("docs/independent_cross_fidelity_holdout.md").read_text(
        encoding="utf-8"
    )
    for token in (
        "main cross-fidelity family：`FAIL`",
        "2.293338 > 2.0",
        "secondary P-Steane family：`PASS`",
        "不进入 sBs 主排名",
        "不得重选",
    ):
        assert token in report
