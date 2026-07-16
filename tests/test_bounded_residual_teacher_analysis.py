from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest

from cnn_fpga.benchmark.bounded_residual_teacher_analysis import (
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    TeacherAnalysisConfig,
    fixed_sequence_library,
    implementation_sha256,
    run_teacher_hidden_control_analysis,
    trace_teacher_hidden,
    validate_production_design,
)
from cnn_fpga.benchmark.bounded_residual_rnn_teacher import (
    load_and_verify_teacher_checkpoint,
)


def _artifact() -> dict[str, object]:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"analysis_half_cycles": 10, "physics_full_cycles": 6}, "cover"),
        ({"probe_training_seed": 1, "probe_evaluation_seed": 1}, "disjoint"),
        ({"probe_ground_probability": 1.0}, "must lie"),
        ({"memory_thresholds": (0.01, 0.10)}, "descending"),
    ),
)
def test_analysis_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        TeacherAnalysisConfig(**kwargs)  # type: ignore[arg-type]


def test_production_design_rejects_short_demo() -> None:
    config = TeacherAnalysisConfig(
        analysis_half_cycles=16,
        physics_full_cycles=2,
        cutoff=4,
        probe_training_trajectories=4,
        probe_evaluation_trajectories=2,
        exponential_decay_grid_size=128,
    )
    with pytest.raises(ValueError, match="analysis_half_cycles"):
        validate_production_design(config)


def test_fixed_sequence_library_has_native_and_explicit_leakage_proxy() -> None:
    sequences = fixed_sequence_library(64)
    assert {
        "all_g",
        "all_e",
        "alternating_ge",
        "alternating_eg",
        "block_ge",
        "block_eg",
        "e_impulse_then_g",
        "g_impulse_then_e",
        "leakage_reset_nominal_proxy",
    } <= set(sequences)
    assert all(len(sequence) == 64 for sequence in sequences.values())
    assert sequences["leakage_reset_nominal_proxy"].count("leak") == 1
    assert all(
        set(sequence) <= {0, 1}
        for name, sequence in sequences.items()
        if not name.startswith("leakage_")
    )


def test_native_trace_rejects_leakage_or_any_third_token() -> None:
    pytest.importorskip("torch")
    model, _ = load_and_verify_teacher_checkpoint()
    with pytest.raises(ValueError, match="only g=0/e=1"):
        trace_teacher_hidden(model, np.asarray([[0, 2, 1]], dtype=np.int64))


def test_committed_analysis_passes_all_gates_and_hashes() -> None:
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


def test_hidden_control_pca_exponential_and_memory_are_complete() -> None:
    payload = _artifact()
    structure = payload["low_dimensional_structure"]
    assert 1 <= structure["hidden_pc95_dimensions"] <= 10
    assert 1 <= structure["control_pc95_dimensions"] <= 15
    assert abs(sum(structure["hidden_pca"]["explained_variance_ratio"]) - 1.0) < 1e-12
    assert abs(
        sum(structure["control_residual_pca"]["explained_variance_ratio"]) - 1.0
    ) < 1e-12
    saturation = payload["exponential_saturation"]
    assert saturation["total_fit_count"] == 30
    assert set(saturation["per_sequence_parameter"]) == {"all_g", "all_e"}
    assert all(
        len(fits) == 15 for fits in saturation["per_sequence_parameter"].values()
    )
    memory = payload["effective_memory"]
    assert len(memory) == 4
    assert all(item["peak_distance"] > 0.0 for item in memory.values())
    assert all(len(item["threshold_crossings"]) == 3 for item in memory.values())


def test_belief_probe_is_trajectory_disjoint_and_not_truth_claim() -> None:
    payload = _artifact()
    probe = payload["belief_state_proxy"]
    split = probe["split"]
    assert split["training_seed"] != split["evaluation_seed"]
    assert probe["hidden_linear_probe"]["training_rows"] == (
        split["training_trajectories"] * payload["config"]["physics_full_cycles"] * 2
    )
    assert probe["hidden_linear_probe"]["evaluation_rows"] == (
        split["evaluation_trajectories"] * payload["config"]["physics_full_cycles"] * 2
    )
    assert np.isfinite(probe["hidden_linear_probe"]["evaluation_r_squared"])
    assert "not hidden physical truth" in probe["target"]
    assert probe["selection_rule"].startswith("none")


def test_conditional_pg_and_leakage_boundaries_are_explicit() -> None:
    payload = _artifact()
    probability = payload["conditional_ground_probability"]
    assert 0.0 <= probability["fixed_minimum"] <= probability["fixed_maximum"] <= 1.0
    assert 0.0 <= probability["probe_minimum"] <= probability["probe_maximum"] <= 1.0
    leakage = payload["leakage_proxy"]
    assert leakage == {
        "teacher_native": False,
        "token_passed_to_teacher": False,
        "policy": "reset hidden and force exactly zero residual for the first post-leakage action",
        "native_leakage_claim_allowed": False,
        "p_g_defined": False,
    }
    assert "student gain retention" in payload["claim_boundary"]["forbidden"]


def test_source_data_has_native_proxy_probe_and_fit_rows() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"]
    assert {row["row_type"] for row in rows} == {
        "fixed_teacher_trace",
        "leakage_ood_proxy_trace",
        "belief_probe_trace",
        "exponential_fit",
    }
    post_leak = [
        row
        for row in rows
        if row["action_source"] == "safe_nominal_after_leakage_proxy"
    ]
    assert len(post_leak) == 1
    assert json.loads(post_leak[0]["physical_residual_json"]) == [0.0] * 15
    assert post_leak[0]["teacher_native"] == "False"


def test_pilot_analysis_executes_but_fails_production_row_gate(tmp_path) -> None:
    pytest.importorskip("torch")
    config = TeacherAnalysisConfig(
        analysis_half_cycles=16,
        physics_full_cycles=2,
        cutoff=4,
        probe_training_trajectories=4,
        probe_evaluation_trajectories=2,
        exponential_decay_grid_size=128,
    )
    result = run_teacher_hidden_control_analysis(
        config,
        artifact_path=tmp_path / "pilot.json",
        source_data_path=tmp_path / "pilot.csv",
        production=False,
    )
    assert result["status"] == "FAIL"
    assert result["gate_summary"]["failed"] == [
        "source_data_contains_fixed_probe_leakage_and_fit_rows"
    ]
