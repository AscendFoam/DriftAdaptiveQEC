from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.phase9_fresh_twin_design_power import (
    CONFIG_PATH,
    PASS_VERDICT,
    build_blueprint,
    build_report,
    main,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_power():
    return build_report(ROOT)


def test_full_design_power_passes_and_selects_smallest_candidate(live_power) -> None:
    report, rows, blueprint = live_power
    assert report["verdict"] == PASS_VERDICT
    assert report["gate_summary"] == {"passed": 22, "total": 22}
    assert report["selected_sample_counts"] == {
        "round_sample_count": 768,
        "trajectory_sample_count": 256,
    }
    assert report["candidate_pair_count"] == 12
    assert report["blueprint"]["gate_count"] == len(blueprint)
    decisions = [row for row in rows if row["row_type"] == "candidate_decision"]
    selected = report["model_selected_sample_counts"]
    index = next(
        index for index, row in enumerate(decisions)
        if row["round_count"] == selected["round_sample_count"]
        and row["trajectory_count"] == selected["trajectory_sample_count"]
    )
    assert decisions[index]["power_gate_pass"] is True
    assert all(row["power_gate_pass"] is not True for row in decisions[:index])


def test_blueprint_has_every_family_cell_and_unique_id(config) -> None:
    blueprint = build_blueprint(config)
    assert len(blueprint) > 1000
    assert len({gate.gate_id for gate in blueprint}) == len(blueprint)
    assert {gate.family for gate in blueprint} == set(config["families"])
    assert any("cutoff/12-16" in gate.gate_id for gate in blueprint)
    assert any("cutoff/16-20" in gate.gate_id for gate in blueprint)
    assert not any("cutoff/8-12" in gate.gate_id for gate in blueprint)
    assert all(gate.margin > 0.0 for gate in blueprint)


def test_power_covers_same_backend_ab_and_every_mutant(live_power, config) -> None:
    _, rows, _ = live_power
    null_cases = {
        row["case"] for row in rows if row["row_type"] == "null_power"
    }
    assert null_cases == {"same_backend_a", "same_backend_b", "ab_zero_effect"}
    mutant_cases = {
        row["case"] for row in rows if row["row_type"] == "alternative_power"
    }
    assert mutant_cases == set(config["alternative_mutants"])


@pytest.mark.parametrize(
    "mutation",
    [
        "historical_access",
        "pilot_margin",
        "seed_overlap",
        "formal_access",
        "weak_alternative",
        "remove_family",
        "max_t",
        "mixed_units",
        "raw_score_primary",
        "delete_cells",
        "cutoff_rescue",
        "lower_floor",
    ],
)
def test_governance_mutations_are_rejected(config, mutation) -> None:
    mutated = copy.deepcopy(config)
    if mutation == "historical_access":
        mutated["historical_policy"]["historical_formal_cell_data_access_allowed"] = True
    elif mutation == "pilot_margin":
        mutated["historical_policy"]["pilot_may_choose"].append("margin")
    elif mutation == "seed_overlap":
        mutated["splits"]["null_backend_b_replica_1"]["start"] = 1040000
    elif mutation == "formal_access":
        mutated["power_model"]["formal_pool_access"] = True
    elif mutation == "weak_alternative":
        mutated["alternative_mutants"]["posterior_temperature"]["effect_ratio"] = 1.01
    elif mutation == "remove_family":
        del mutated["families"]["reset_leakage"]
    elif mutation == "max_t":
        mutated["statistical_procedure"]["global_test"] = "global_max_t"
    elif mutation == "mixed_units":
        mutated["statistical_procedure"]["mixed_unit_vector_max"] = True
    elif mutation == "raw_score_primary":
        mutated["statistical_procedure"]["raw_log_evidence"] = "primary"
    elif mutation == "delete_cells":
        mutated["formal_matrix"]["all_cells_required"] = False
    elif mutation == "cutoff_rescue":
        mutated["formal_matrix"]["primary_cutoff_increments"] = [[8, 12]]
    elif mutation == "lower_floor":
        mutated["power_model"]["unmodeled_physics_conservative_floor"][
            "round_sample_count"
        ] = 256
    with pytest.raises((TypeError, ValueError)):
        validate_config(mutated)


def test_repeated_build_is_bit_deterministic(live_power) -> None:
    first, first_rows, first_blueprint = live_power
    second, second_rows, second_blueprint = build_report(ROOT)
    assert first == second
    assert first_rows == second_rows
    assert first_blueprint == second_blueprint


def test_cli_overrides_are_forbidden() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--round-count", "1"])
    assert raised.value.code == 2
