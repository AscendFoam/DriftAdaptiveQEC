from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.run_length_fsm_baseline import (
    RUN_LENGTH_DESCRIPTOR,
    RunLengthValidationConfig,
    _evaluate_trace,
    _implementation_sha256,
    _make_trace,
    event_control_cost,
    event_scenarios,
)
from cnn_fpga.runtime.run_length_fsm import (
    FALLBACK,
    LEAKAGE_HOLD,
    NORMAL,
    X_RECOVERY,
    Z_RECOVERY,
    RunLengthFSMConfig,
)


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_2_5_run_length_fsm_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_2_5_run_length_fsm_source_data.csv"


def _payload() -> dict[str, object]:
    return json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))


def test_descriptor_is_observed_only_event_controller_not_ler_claim() -> None:
    assert RUN_LENGTH_DESCRIPTOR.hidden_truth_inputs == ()
    assert RUN_LENGTH_DESCRIPTOR.logical_error_metric is False
    assert RUN_LENGTH_DESCRIPTOR.fixed_point_or_rtl is False
    assert RUN_LENGTH_DESCRIPTOR.target_hardware_measured is False
    assert "event_control_cost" in RUN_LENGTH_DESCRIPTOR.primary_metric


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"training_seeds": (1, 2)}, "at least three"),
        ({"evaluation_seeds": (1, 2, 3, 4, 5)}, "at least six"),
        (
            {"training_seeds": (1, 2, 3), "evaluation_seeds": (3, 4, 5, 6, 7, 8)},
            "disjoint",
        ),
        ({"training_cycles": 255}, "at least 256"),
        ({"evaluation_cycles": 511}, "at least 512"),
        ({"e_enter_grid": (1, 2)}, "memoryless comparator"),
        ({"e_enter_grid": (2, 8)}, "3-bit"),
        ({"leakage_enter_grid": ()}, "nonempty"),
    ],
)
def test_validation_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        RunLengthValidationConfig(**kwargs)  # type: ignore[arg-type]


def test_event_cost_matrix_has_exact_oracle_zero_and_penalizes_wrong_axis() -> None:
    for mode in (NORMAL, X_RECOVERY, Z_RECOVERY, LEAKAGE_HOLD, FALLBACK):
        assert event_control_cost(mode, mode) == 0.0
    assert event_control_cost(X_RECOVERY, Z_RECOVERY) > event_control_cost(
        X_RECOVERY, FALLBACK
    )
    assert event_control_cost(LEAKAGE_HOLD, NORMAL) > event_control_cost(
        LEAKAGE_HOLD, FALLBACK
    )
    with pytest.raises(ValueError, match="unknown"):
        event_control_cost("truth", NORMAL)


def test_trace_generation_and_real_fsm_replay_are_seed_deterministic() -> None:
    scenario = event_scenarios()[3]
    first = _make_trace(scenario, 1234, 512, 3)
    second = _make_trace(scenario, 1234, 512, 3)
    config = RunLengthFSMConfig(
        e_enter_run=2,
        leakage_enter_run=1,
        leakage_clear_run=2,
        fallback_clear_run=2,
    )
    first_row, first_modes = _evaluate_trace(first, config)
    second_row, second_modes = _evaluate_trace(second, config)

    assert first.trace_sha256 == second.trace_sha256
    assert first.truth_modes == second.truth_modes
    assert first_modes == second_modes
    assert first_row == second_row
    assert first_row["fsm_final_bank_version"] == first_row["run_length_fsm_bank_writes"]


def test_production_artifact_is_source_bound_non_demo_and_all_gates_pass() -> None:
    payload = _payload()
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_sha256()
    assert payload["aggregate"]["evaluation_cycles"] == 384_000
    assert payload["aggregate"]["evaluation_traces"] == 32
    assert payload["aggregate"]["source_data_rows"] == 32
    assert payload["training_selection"]["training_cycles"] == 49_152
    assert payload["training_selection"]["training_fsm_replay_cycles"] == 1_179_648
    assert len(payload["training_selection"]["grid"]) == 24
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == 15


def test_training_selection_excludes_degenerate_memoryless_threshold() -> None:
    payload = _payload()
    selection = payload["training_selection"]
    assert selection["evaluation_truth_used"] is False
    assert selection["selected_config"]["e_enter_run"] >= 2
    assert all(row["e_enter_run"] >= 2 for row in selection["grid"])
    assert {row["training_traces"] for row in selection["grid"]} == {12}


def test_source_data_recomputes_versions_and_keeps_same_trace_comparators() -> None:
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 32
    assert len({row["trace_sha256"] for row in rows}) == 32
    assert {row["scenario_id"] for row in rows} == {
        scenario.scenario_id for scenario in event_scenarios()
    }
    for row in rows:
        assert int(row["fsm_final_bank_version"]) == int(
            row["run_length_fsm_bank_writes"]
        )
        assert int(row["fsm_bank_conflicts"]) == 0
        assert row["fsm_corrections_finite"] == "True"
        assert float(row["truth_oracle_event_plus_write_cost"]) <= float(
            row["run_length_fsm_event_plus_write_cost"]
        )
        for controller in (
            "static_safe_normal",
            "memoryless_event",
            "run_length_fsm",
            "truth_oracle",
        ):
            assert 0.0 <= float(row[f"{controller}_action_accuracy"]) <= 1.0


def test_conflict_probe_and_claim_boundary_are_explicit() -> None:
    payload = _payload()
    probe = payload["parameter_bank_conflict_probe"]
    assert probe["modes"] == [FALLBACK, FALLBACK, FALLBACK, X_RECOVERY]
    assert probe["conflicts"] == [True, True, True, False]
    assert probe["local_safe_rom"] == [True, True, True, False]
    assert probe["final_active_mode"] == X_RECOVERY
    assert probe["final_version"] == 2
    assert "logical-error-rate gain" in payload["claim_boundary"]["forbidden"]
