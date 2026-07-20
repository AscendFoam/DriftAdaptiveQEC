from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import randomized_model_mismatch as audit
from physics.differentiable_sbs_trajectory import DifferentiableSBSConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/t5_4_6_randomized_model_mismatch.json"
SOURCE = ROOT / "docs/t5_4_6_randomized_model_mismatch_source_data.csv"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rehash(mutated: dict) -> dict:
    mutated["contract_sha256"] = audit._canonical_sha256(
        audit._contract_view(mutated)
    )
    return mutated


def test_formal_artifact_is_semantically_valid(report: dict) -> None:
    assert report["status"] == "PASS"
    assert report["verdict"] == (
        "EVIDENCE_COMPLETE_QUALIFIED_STUDENT_BRANCH_RETAINED"
    )
    assert report["gate_summary"] == {"passed": 19, "total": 19}
    assert audit.validate_artifact(report) == ()


def test_all_parent_and_implementation_bindings_are_current(report: dict) -> None:
    assert len(report["parent_bindings"]) == 9
    assert all(row["machine_pass"] for row in report["parent_bindings"])
    for row in report["parent_bindings"] + report["implementation_bindings"]:
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert row["sha256"] == audit._sha256(path)


def test_production_design_rejects_smoke_scale() -> None:
    audit.validate_production_design(audit.RandomizedMismatchConfig())
    with pytest.raises(ValueError, match=">=8"):
        audit.validate_production_design(
            audit.RandomizedMismatchConfig(physical_cells_per_family=1)
        )
    with pytest.raises(ValueError, match="cutoff"):
        audit.validate_production_design(
            audit.RandomizedMismatchConfig(physical_cutoff=10)
        )
    with pytest.raises(ValueError, match="lifetime fitting"):
        audit.RandomizedMismatchConfig(physical_full_cycles=1)


def test_cli_pilot_uses_small_config_and_cannot_overwrite_formal_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_run(config, *, artifact_path, source_data_path, production):
        captured.update(
            config=config,
            artifact_path=Path(artifact_path),
            source_data_path=Path(source_data_path),
            production=production,
        )
        return {
            "status": "FAIL",
            "verdict": "INCOMPLETE_EVIDENCE_GATE_FAILURE",
            "gate_summary": {"passed": 14, "total": 19},
            "branch_decision": {"output_branch": "qualified_student_retention"},
        }

    monkeypatch.setattr(audit, "run_randomized_model_mismatch", fake_run)
    assert audit.main(["--pilot", "--device", "cpu"]) == 2
    assert captured["production"] is False
    assert captured["config"].physical_cells_per_family == 1
    assert captured["config"].component_cells_per_family == 2
    assert captured["config"].drift_cells == 2
    assert captured["config"].physical_cutoff == 6
    assert captured["artifact_path"] == Path(".tmp_t546/pilot.json")
    assert captured["source_data_path"] == Path(".tmp_t546/pilot.csv")
    assert captured["artifact_path"] != audit.DEFAULT_ARTIFACT


def test_registry_contains_64_distinct_parent_disjoint_cells(report: dict) -> None:
    registry = report["mismatch_registry"]
    assert len(registry) == 64
    assert len({(row["lane"], row["cell_id"]) for row in registry}) == 64
    assert len({row["vector_sha256"] for row in registry}) == 64
    assert report["seed_audit"]["evaluation_seed_count"] == 64
    assert report["seed_audit"]["overlap_with_parent_seeds"] == []
    assert registry == audit._mismatch_registry(report)


def test_physical_cells_cover_full_vectors_phase_noise_and_timing(report: dict) -> None:
    cells = report["physical_lane"]["cells"]
    assert len(cells) == 32
    assert {row["family"] for row in cells} == set(audit.PHYSICAL_FAMILIES)
    assert all(
        sum(row["family"] == family for row in cells) == 8
        for family in audit.PHYSICAL_FAMILIES
    )
    biased = [
        row
        for row in cells
        if row["family"] in {"gate_bias_vector", "compound_physical"}
    ]
    assert len(biased) == 16
    assert all(row["gate_bias_nonzero_dimensions"] == 15 for row in biased)
    assert len({tuple(row["gate_bias"]) for row in biased}) == 16
    dephased = [
        row
        for row in cells
        if row["family"] in {"cavity_dephasing", "compound_physical"}
    ]
    assert all(80.0 <= row["cavity_tphi_us"] <= 1000.0 for row in dephased)
    timed = [
        row
        for row in cells
        if row["family"] in {"unseen_timing_dynamics", "compound_physical"}
    ]
    assert all(row["timing_total_ns"] == 5000 for row in cells)
    assert len({tuple(row["timing_durations_ns"]) for row in timed}) == 16


def test_every_physical_cell_executes_all_three_paired_strategies(
    report: dict,
) -> None:
    lane = report["physical_lane"]
    assert len(lane["strategy_rows"]) == 96
    for cell in lane["cells"]:
        rows = [
            row for row in lane["strategy_rows"] if row["cell_id"] == cell["cell_id"]
        ]
        assert {row["strategy"] for row in rows} == set(audit.PHYSICAL_STRATEGIES)
        assert all(row["seed"] == cell["seed"] for row in rows)
        assert all(set(row) >= {"matched", "mismatch"} for row in rows)


def test_physical_channels_and_outputs_remain_numerically_physical(
    report: dict,
) -> None:
    diagnostics = report["physical_lane"]["cell_diagnostics"]
    assert len(diagnostics) == 32
    assert max(row["idle_completeness_max"] for row in diagnostics) < 1.0e-10
    assert min(
        row["phase_kernel_minimum_eigenvalue"]
        for row in diagnostics
        if row["phase_kernel_minimum_eigenvalue"] is not None
    ) >= -1.0e-8
    for row in report["physical_lane"]["strategy_rows"]:
        for condition in ("matched", "mismatch"):
            metrics = row[condition]
            assert metrics["maximum_trace_error"] < 1.0e-7
            assert metrics["maximum_hermiticity_error"] < 1.0e-7
            assert metrics["minimum_final_eigenvalue"] >= -1.0e-8
            assert np.isfinite(metrics["selection_score"])


def test_phase_diffusion_is_applied_as_a_cptp_schur_kernel() -> None:
    torch = pytest.importorskip("torch")
    config = DifferentiableSBSConfig(
        cutoff=6,
        full_cycles=2,
        batch_size=1,
        device="cpu",
        real_dtype="float64",
    )
    nominal = audit.RandomizedMismatchSimulator(
        config, gate_bias=(0.0,) * 15, cavity_tphi_us=None
    )
    dephased = audit.RandomizedMismatchSimulator(
        config, gate_bias=(0.0,) * 15, cavity_tphi_us=100.0
    )
    state = nominal._initial_joint_density()
    reference = nominal._apply_idle(state, "layer_2")
    observed = dephased._apply_idle(state, "layer_2")
    duration = dephased._phase_duration_us["layer_2"]
    kernel = torch.exp(
        -0.5
        * duration
        / dephased.cavity_tphi_us
        * dephased._number_difference_sq
    )
    expected = (
        reference.reshape(1, 6, 2, 6, 2)
        * kernel[None, :, None, :, None]
    ).reshape_as(reference)
    assert torch.max(torch.abs(observed - expected)).item() < 1.0e-12
    assert abs(torch.trace(observed[0]).real.item() - 1.0) < 1.0e-12
    assert dephased.phase_kernel_minimum_eigenvalue() >= -1.0e-10


def test_timing_randomizer_preserves_total_and_changes_allocation() -> None:
    profiles = [audit._duration_profile(np.random.default_rng(seed)) for seed in range(8)]
    assert all(profile.half_cycle_duration_ns == 5000 for profile in profiles)
    assert len(
        {tuple(phase.duration_ns for phase in profile.phases) for profile in profiles}
    ) == 8
    assert all(
        min(phase.duration_ns for phase in profile.phases) >= 20
        for profile in profiles
    )


def test_branch_is_recomputed_from_raw_strategy_scores(report: dict) -> None:
    lane = report["physical_lane"]
    rows, summary = audit._retention_evidence(
        lane["cells"], lane["strategy_rows"], report["config"]
    )
    assert rows == lane["retention_rows"]
    assert summary == lane["retention_summary"]
    assert summary["qualifying_teacher_gain_cells"] == 32
    assert summary["positive_teacher_gain_fraction"] == 1.0
    assert summary["retention_median"] >= 0.99
    assert summary["retention_minimum"] < summary["retention_median"]
    assert report["branch_decision"] == audit._branch_decision(lane, report["config"])


def test_readout_lane_exercises_every_row_of_all_random_4x3_matrices(
    report: dict,
) -> None:
    lane = report["readout_lane"]
    assert lane["cell_count"] == lane["unique_matrix_hashes"] == 8
    for row in lane["rows"]:
        matrix = np.asarray(row["confusion_matrix"])
        full = row["full_matrix_audit"]
        counts = np.asarray(full["counts"])
        assert matrix.shape == counts.shape == (4, 3)
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert np.all(counts.sum(axis=1) == full["trials_per_hidden_state"])
        assert full["all_four_hidden_rows_exercised"] is True
        assert full["full_matrix_calibrated"] is True
        assert row["confusion_calibrated"] is True
    assert max(
        row["full_matrix_audit"]["maximum_absolute_entry_error"]
        for row in lane["rows"]
    ) < 0.005


def test_direct_full_matrix_audit_is_deterministic_and_rejects_bad_rows() -> None:
    matrix = audit._random_confusion_matrix(np.random.default_rng(546))
    first = audit._audit_full_confusion_matrix(matrix, 77, 4096)
    second = audit._audit_full_confusion_matrix(matrix, 77, 4096)
    assert first == second
    assert first["full_matrix_calibrated"] is True
    bad = copy.deepcopy(matrix)
    bad[0][0] += 0.1
    with pytest.raises(ValueError, match="stochastic"):
        audit._audit_full_confusion_matrix(bad, 77, 4096)


def test_leakage_and_reset_lanes_use_random_rates_and_auditable_trials(
    report: dict,
) -> None:
    rows = report["leakage_reset_lane"]["rows"]
    assert len(rows) == 16
    assert {row["family"] for row in rows} == set(audit.LEAKAGE_FAMILIES)
    for family in audit.LEAKAGE_FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        assert len(selected) == 8
        assert len({row["target_rate"] for row in selected}) == 8
    for row in rows:
        assert row["calibration_trial_count"] > 0
        assert row["channel_calibrated"] is True
        assert abs(row[row["target_metric"]] - row["target_rate"]) <= row[
            "calibration_tolerance"
        ]
        assert "/" in row["calibration_trial_count_source"]


def test_drift_lane_uses_all_unseen_dynamics_and_frozen_decoders(
    report: dict,
) -> None:
    lane = report["drift_lane"]
    assert len(lane["rows"]) == 8
    assert set(lane["dynamics_families"]) == {
        "chirped_sinusoid",
        "random_telegraph",
        "ramp_burst",
    }
    assert lane["frozen_parent_binding"]["hyperparameters_reselected_on_ood"] is False
    assert lane["frozen_parent_binding"]["static_parameters_refit_on_ood"] is False
    assert len({row["trace_sha256"] for row in lane["rows"]}) == 8
    assert all(row["shared_trace_for_all_methods"] for row in lane["rows"])
    for row in lane["rows"]:
        for method in ("standard", "static", "window", "ewma", "kalman", "oracle"):
            assert 0.0 <= row[f"{method}_error_rate"] <= 1.0


@pytest.mark.parametrize(
    "dynamics",
    ("chirped_sinusoid", "random_telegraph", "ramp_burst"),
)
def test_random_drift_generators_are_stateful_and_bounded(dynamics: str) -> None:
    states, parameters = audit._random_drift_states(1234, 48, dynamics=dynamics)
    assert len(states) == 48
    assert parameters["dynamics"] == dynamics
    assert len({state.mu_q for state in states}) >= (
        2 if dynamics == "random_telegraph" else 3
    )
    assert min(state.sigma_q for state in states) > 0.0
    assert max(abs(state.rho) for state in states) < 1.0
    assert all(0.0 <= state.p_outlier <= 1.0 for state in states)


def test_no_cross_lane_score_or_hardware_claim_is_promoted(report: dict) -> None:
    assert report["cross_lane_aggregate"] is None
    assert report["global_ranking"] is None
    boundary = report["claim_boundary"]
    assert boundary["physical_memory_ler_established"] is False
    assert boundary["device_calibrated"] is False
    assert boundary["hardware_measured"] is False
    assert boundary["experimental_claim"] is False


def test_source_csv_is_complete_and_byte_bound(report: dict) -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["source_data"]["row_count"] == 273
    assert report["source_data"]["csv_sha256"] == audit._sha256(SOURCE)
    assert {row["row_type"] for row in rows} >= {
        "mismatch_registry",
        "physical_strategy_cell",
        "physical_retention",
        "readout_cell",
        "leakage_reset_cell",
        "drift_cell",
        "branch_decision",
        "gate",
    }
    gate_rows = [row for row in rows if row["row_type"] == "gate"]
    assert len(gate_rows) == 19
    assert all(row["value"] == "1" for row in gate_rows)


@pytest.mark.parametrize(
    "mutation",
    (
        "fixed_bias_vector",
        "drop_physical_cell",
        "duplicate_strategy",
        "inflate_student_score",
        "rewrite_branch_threshold",
        "skip_hidden_readout_row",
        "fake_reset_calibration",
        "refit_drift_parent",
        "cross_lane_score",
        "hardware_claim",
    ),
)
def test_validator_rejects_randomized_mismatch_shortcuts(
    report: dict, mutation: str
) -> None:
    changed = copy.deepcopy(report)
    if mutation == "fixed_bias_vector":
        cells = changed["physical_lane"]["cells"]
        source = next(row for row in cells if row["family"] == "gate_bias_vector")
        target = next(
            row
            for row in cells
            if row["family"] == "gate_bias_vector" and row is not source
        )
        target["gate_bias"] = source["gate_bias"]
    elif mutation == "drop_physical_cell":
        changed["physical_lane"]["cells"].pop()
    elif mutation == "duplicate_strategy":
        changed["physical_lane"]["strategy_rows"][0]["strategy"] = "student"
    elif mutation == "inflate_student_score":
        row = next(
            row
            for row in changed["physical_lane"]["strategy_rows"]
            if row["strategy"] == "student"
        )
        row["mismatch"]["selection_score"] += 0.2
    elif mutation == "rewrite_branch_threshold":
        changed["branch_decision"]["thresholds"]["minimum_retention_median"] = 0.01
    elif mutation == "skip_hidden_readout_row":
        changed["readout_lane"]["rows"][0]["full_matrix_audit"]["counts"][3][0] += 1
    elif mutation == "fake_reset_calibration":
        row = next(
            row
            for row in changed["leakage_reset_lane"]["rows"]
            if row["family"] == "random_reset_failure"
        )
        row["empirical_reset_failure_probability"] = 0.0
        row["channel_calibrated"] = True
    elif mutation == "refit_drift_parent":
        changed["drift_lane"]["frozen_parent_binding"][
            "hyperparameters_reselected_on_ood"
        ] = True
    elif mutation == "cross_lane_score":
        changed["cross_lane_aggregate"] = 0.99
        changed["global_ranking"] = ["student", "kalman"]
    elif mutation == "hardware_claim":
        changed["claim_boundary"]["hardware_measured"] = True
        changed["claim_boundary"]["experimental_claim"] = True
    errors = audit.validate_artifact(_rehash(changed), check_files=False)
    assert errors
    assert any("stored gates" in error or "gate recomputation" in error for error in errors)
