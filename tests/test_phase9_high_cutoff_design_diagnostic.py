from __future__ import annotations

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_high_cutoff_design_diagnostic as subject
from cnn_fpga.benchmark.phase9_paired_cluster_uq import NormUCB


def _fake_ucb(left, right, **kwargs) -> NormUCB:
    count = len(left)
    return NormUCB(
        estimate=0.01,
        raw_radius=0.02,
        calibrated_radius=0.02,
        quantization_bound=float(
            np.mean(kwargs.get("quantization_bounds", np.zeros(count)))
        ),
        upper_bound=0.03,
        confidence=float(kwargs["confidence"]),
        multiplier_replicates=int(kwargs["multiplier_replicates"]),
        cluster_count=count,
        calibration_factor=float(kwargs["calibration_factor"]),
        seed=int(kwargs["seed"]),
    )


def _fixture():
    states = ("0", "1", "+", "-", "+i", "-i")
    rows = []
    densities = {}
    for cutoff in (2, 3):
        for backend in ("A", "B"):
            for position in range(72):
                state = states[position % 6]
                for round_index in range(12):
                    row_id = (
                        f"c{cutoff}-{backend}-p{position}-r{round_index}"
                    )
                    terminal = round_index == 11
                    rows.append(
                        {
                            "row_id": row_id,
                            "cutoff": cutoff,
                            "scenario": "step",
                            "backend": backend,
                            "logical_label": state,
                            "seed_position": position,
                            "round_index": round_index,
                            "terminal_round": terminal,
                            "mean_photon": 0.1 + 0.001 * position,
                            "level_g": 0.8,
                            "level_e": 0.15,
                            "level_f": 0.05,
                            "logical_survival": 0.9,
                            "density_quantization_trace_distance_bound": 1e-7,
                        }
                    )
                    if terminal:
                        densities[row_id] = np.eye(3 * cutoff) / (3 * cutoff)
    config = {
        "cutoffs": [2, 3],
        "scenario_names": ["step"],
        "logical_state_schedule": list(states),
        "stage_partition": {"step": {"all": list(range(12))}},
        "diagnostic_contract": {
            "confidence": 0.95,
            "multiplier_replicates": 1999,
            "multiplier_seed_namespace": 1320000,
            "calibration_factor_source": {"required_factor": 1.0},
            "margins": {
                "ab_terminal_density_trace_distance": 0.15,
                "ab_terminal_mean_photon_difference": 0.08,
                "ab_terminal_level_probability_l1": 0.1,
                "ab_terminal_logical_survival_difference": 0.1,
                "cutoff_terminal_density_trace_distance": 0.1,
                "cutoff_terminal_mean_photon_difference": 0.08,
                "cutoff_terminal_level_probability_l1": 0.1,
                "cutoff_terminal_logical_survival_difference": 0.1,
            },
        },
    }
    return config, rows, densities


def test_diagnostic_enforces_state_stage_backend_and_cutoff_iut(monkeypatch):
    config, rows, densities = _fixture()
    monkeypatch.setattr(subject, "paired_density_trace_ucb", _fake_ucb)
    monkeypatch.setattr(subject, "paired_vector_norm_ucb", _fake_ucb)
    results = subject.evaluate_diagnostics(config, rows, densities)
    assert len(results) == 96
    assert len({row["gate_id"] for row in results}) == 96
    assert {row["logical_state"] for row in results} == {
        "0",
        "1",
        "+",
        "-",
        "+i",
        "-i",
    }
    assert {row["contrast"] for row in results} == {
        "same_cutoff_ab",
        "within_backend_cutoff",
    }
    assert all(row["cluster_count"] == 12 for row in results)
    assert all(row["design_pilot_only"] is True for row in results)


def test_missing_state_cluster_fails_closed(monkeypatch):
    config, rows, densities = _fixture()
    removed = rows.pop()
    densities.pop(removed["row_id"], None)
    monkeypatch.setattr(subject, "paired_density_trace_ucb", _fake_ucb)
    monkeypatch.setattr(subject, "paired_vector_norm_ucb", _fake_ucb)
    with pytest.raises(ValueError, match="round coverage"):
        subject.evaluate_diagnostics(config, rows, densities)


def test_cutoff_embedding_preserves_trace_and_top_left_block() -> None:
    lower = np.eye(6) / 6
    embedded = subject._embed_density(lower, 2, 3)
    assert embedded.shape == (9, 9)
    assert np.trace(embedded) == pytest.approx(1.0)
    assert np.array_equal(embedded[:6, :6], lower)
    assert np.count_nonzero(embedded[6:, :]) == 0


def test_multiplier_namespace_is_disjoint_and_stable() -> None:
    first = subject._seed(1320000, "gate-a")
    repeat = subject._seed(1320000, "gate-a")
    other = subject._seed(1320000, "gate-b")
    assert first == repeat
    assert first != other
    assert first >> 64 == 1320000
