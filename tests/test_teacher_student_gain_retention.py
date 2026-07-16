from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.teacher_student_gain_retention import (
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    DistilledStudentTorchPolicy,
    GainRetentionConfig,
    _retention_bootstrap,
    implementation_sha256,
    run_teacher_student_gain_retention,
    validate_production_design,
)
from cnn_fpga.control.low_dimensional_recurrence import (
    LowDimensionalRecurrenceArtifact,
)


STUDENT = Path("docs/t4_4_3_low_dimensional_student.json")


def _artifact() -> dict[str, object]:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"exact_control_oracle_cycles": 3}, "exactly two"),
        ({"primary_seeds": (1, 1)}, "unique"),
        ({"primary_seeds": (1, 2), "confirmation_seeds": (2, 3)}, "disjoint"),
        ({"minimum_gain_retention_fraction": 0.0}, "must lie"),
        ({"minimum_gain_retention_ci_lower": 1.1}, "must lie"),
        ({"maximum_teacher_student_pg_difference": 1.0}, "must lie"),
        ({"device": "tpu"}, "cpu or cuda"),
    ),
)
def test_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        GainRetentionConfig(**kwargs)  # type: ignore[arg-type]


def test_production_design_rejects_demo_and_changed_threshold() -> None:
    with pytest.raises(ValueError, match="full_cycles"):
        validate_production_design(
            GainRetentionConfig(
                full_cycles=2,
                cutoff=4,
                confirmation_cutoff=4,
                primary_batch_size=2,
                confirmation_batch_size=2,
                primary_seeds=(1, 2),
                confirmation_seeds=(3, 4),
                bootstrap_repetitions=100,
                device="cpu",
            )
        )
    with pytest.raises(ValueError, match="point retention threshold"):
        validate_production_design(
            GainRetentionConfig(minimum_gain_retention_fraction=0.95)
        )


def test_nonpositive_bootstrap_gain_is_retained_as_undefined_not_dropped() -> None:
    def evaluation(values: tuple[float, float]) -> dict[str, object]:
        return {
            "per_seed": [
                {
                    "fidelity": {
                        "normalized_auc": value,
                        "effective_lifetime_cycles": value,
                    },
                    "logical_z": {
                        "normalized_auc": value,
                        "effective_lifetime_cycles": value,
                    },
                }
                for value in values
            ]
        }

    result = _retention_bootstrap(
        evaluation((0.5, 0.5)),
        evaluation((0.4, 0.6)),
        evaluation((0.5, 0.6)),
        "selection_score",
        seed=1,
        repetitions=100,
    )
    assert result["defined"] is False
    assert result["point_retention_fraction"] is None
    assert result["positive_teacher_gain_bootstrap_fraction"] == 0.0


def test_student_torch_adapter_returns_raw_head_for_simulator_mapping() -> None:
    torch = pytest.importorskip("torch")
    artifact = LowDimensionalRecurrenceArtifact.from_dict(
        json.loads(STUDENT.read_text(encoding="utf-8"))
    )
    model = DistilledStudentTorchPolicy(
        artifact, device="cpu", dtype=torch.float64
    )
    history = torch.tensor(((0, 1, 1, 0),), dtype=torch.int64)
    raw = model(history, 4).detach().numpy()[0]
    state = np.asarray(artifact.initial_state)
    decays = np.asarray(artifact.outcome_decays)
    saturations = np.asarray(artifact.outcome_saturations)
    for outcome in history.numpy()[0]:
        state = decays[outcome] * state + (1.0 - decays[outcome]) * saturations[outcome]
    expected = np.asarray(artifact.output_weights) @ state + np.asarray(artifact.output_bias)
    np.testing.assert_allclose(raw, expected, rtol=0.0, atol=2.0e-15)


def test_committed_artifact_passes_every_gate_and_is_source_bound() -> None:
    payload = _artifact()
    gates = payload["gates"]
    assert payload["status"] == "PASS"
    assert isinstance(gates, dict) and len(gates) >= 17 and all(gates.values())
    assert payload["gate_summary"] == {
        "passed": len(gates),
        "total": len(gates),
        "failed": [],
    }
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()


def test_ten_cycle_lanes_keep_every_mf_agent_and_new_paired_seeds() -> None:
    payload = _artifact()
    config = payload["config"]
    primary = payload["stochastic_ten_cycle"]["primary"]
    confirmation = payload["stochastic_ten_cycle"]["confirmation"]
    assert primary["full_cycles"] == confirmation["full_cycles"] == 10
    assert primary["seeds"] == config["primary_seeds"]
    assert confirmation["seeds"] == config["confirmation_seeds"]
    assert set(primary["seeds"]).isdisjoint(confirmation["seeds"])
    for lane in (primary, confirmation):
        assert lane["mf_all_agents"]["agent_count"] == 5
        assert len(lane["mf_all_agents"]["agents"]) == 5
        assert {agent["strategy"] for agent in lane["mf_all_agents"]["agents"]} == {
            "mf_agent_101",
            "mf_agent_211",
            "mf_agent_307",
            "mf_agent_401",
            "mf_agent_503",
        }


def test_stochastic_retention_point_and_confidence_gates_are_ninety_percent() -> None:
    payload = _artifact()
    threshold = payload["retention_threshold"]
    assert threshold["point_fraction"] == threshold["paired_bootstrap_ci_lower"] == 0.90
    assert threshold["frozen_before_physical_evaluation"] is True
    for lane in payload["stochastic_retention"].values():
        assert set(lane) == {
            "selection_score",
            "fidelity_effective_lifetime_cycles",
            "logical_z_effective_lifetime_cycles",
        }
        for value in lane.values():
            assert value["defined"]
            assert value["positive_teacher_gain_bootstrap_fraction"] == 1.0
            assert value["point_retention_fraction"] >= 0.90
            assert value["ci_95"][0] >= 0.90


def test_exact_lane_is_two_cycle_complete_and_never_extrapolates_oracle() -> None:
    payload = _artifact()
    assert "control_oracle" not in payload["stochastic_ten_cycle"]["primary"]
    assert "control_oracle" not in payload["stochastic_ten_cycle"]["confirmation"]
    for cutoff, lane in payload["exact_two_cycle"].items():
        assert int(cutoff) in (12, 16)
        assert lane["full_cycles"] == 2
        assert lane["control_oracle_is_horizon_bound"] is True
        assert lane["control_oracle"]["branch_count"] == 16
        assert lane["mf_all_agents"]["agent_count"] == 5
        assert abs(lane["control_oracle"]["trajectory_probability_sum"] - 1.0) < 1e-12
        for value in payload["exact_retention"][cutoff].values():
            assert value["defined"]
            assert value["retention_fraction"] >= 0.90


def test_pg_e_burden_bounds_and_costs_are_not_hardware_claims() -> None:
    payload = _artifact()
    maximum = payload["config"]["maximum_teacher_student_pg_difference"]
    differences = payload["teacher_student_pg_absolute_difference"]
    assert all(value <= maximum for value in differences["stochastic"].values())
    assert all(value <= maximum for value in differences["exact"].values())
    leakage = payload["leakage_burden"]
    assert leakage["native_multilevel_leakage_available"] is False
    assert leakage["reported_value"] is None
    assert "never labelled leakage" in leakage["e_burden_proxy"]
    burden = payload["burden_summary"]
    assert set(burden["stochastic"]) == {"primary", "confirmation"}
    assert set(burden["exact"]) == {"12", "16"}
    assert all(
        row["multilevel_leakage_events"] is None
        and 0.0 <= row.get("observed_e_fraction", row.get("expected_e_fraction")) <= 1.0
        for family in burden.values()
        for lane in family.values()
        for row in lane.values()
    )
    costs = {row["strategy"]: row for row in payload["costs"]}
    assert costs["distilled_student"]["stored_scalars"] == 95
    assert costs["distilled_student"]["persistent_state_scalars"] == 4
    assert costs["distilled_student"]["analytic_macs_per_half_cycle"] == 87
    assert costs["fresh_gru_teacher"]["stored_scalars"] == 72_853
    assert costs["finite_horizon_control_oracle"]["deployable"] is False
    assert all(
        lane["distilled_student"]["maximum_action_bound_violation"] == 0.0
        for lane in payload["exact_two_cycle"].values()
    )


def test_source_data_contains_all_seed_branch_retention_and_cost_rows() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"]
    assert len(rows) == payload["source_data"]["expected_row_count"]
    assert {row["row_type"] for row in rows} == {
        "stochastic_seed_summary",
        "exact_branch",
        "retention_gate",
        "cost_summary",
    }
    assert sum(row["row_type"] == "stochastic_seed_summary" for row in rows) == 108
    assert sum(row["row_type"] == "exact_branch" for row in rows) == 320
    assert sum(row["row_type"] == "retention_gate" for row in rows) == 14
    assert sum(row["row_type"] == "cost_summary" for row in rows) == 6


def test_pilot_executes_and_fails_closed_without_production_horizon(tmp_path) -> None:
    pytest.importorskip("torch")
    config = GainRetentionConfig(
        full_cycles=2,
        cutoff=4,
        confirmation_cutoff=4,
        primary_batch_size=2,
        confirmation_batch_size=2,
        primary_seeds=(94401, 94403),
        confirmation_seeds=(94411, 94417),
        bootstrap_repetitions=100,
        device="cpu",
    )
    result = run_teacher_student_gain_retention(
        config,
        artifact_path=tmp_path / "pilot.json",
        source_data_path=tmp_path / "pilot.csv",
        production=False,
    )
    assert result["status"] == "FAIL"
    assert not result["gates"][
        "ten_cycle_lane_retains_standard_all_five_mf_teacher_handcrafted_and_student"
    ]
    assert result["gates"][
        "source_data_contains_every_seed_branch_retention_and_cost_row"
    ]
    assert "multilevel leakage robustness" in result["claim_boundary"]["forbidden"]
