from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_paired_cluster_uq_calibration as subject


def _family(**overrides):
    value = {
        "left_noise_weight": 0.4,
        "right_noise_weight": 0.4,
        "rare_probability": 1.0,
        "coherent_unitary": False,
        "power_primary": True,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "family",
    [
        _family(),
        _family(left_noise_weight=0.25, right_noise_weight=0.75),
        _family(rare_probability=0.1, coherent_unitary=True),
    ],
)
@pytest.mark.parametrize("truth", [0.0, 0.05, 0.1, 0.12])
def test_physical_density_trial_is_psd_trace_one_and_analytic(family, truth):
    left, right, analytic = subject.physical_density_trial(
        dimension=8,
        count=24,
        true_distance=truth,
        family=family,
        seed=8100,
    )
    assert analytic == pytest.approx(truth, abs=1e-12)
    for stack in (left, right):
        assert np.max(np.abs(np.trace(stack, axis1=1, axis2=2) - 1.0)) < 1e-12
        assert np.min(np.linalg.eigvalsh(stack)) > -1e-12
        assert np.max(np.abs(stack - np.swapaxes(stack.conj(), 1, 2))) < 1e-12


def test_seed_addressing_is_order_stable_and_namespace_separated() -> None:
    first = subject._seed(1270000, "calibration", "family", 16, 64, 0.0, 1)
    second = subject._seed(1270000, "calibration", "family", 16, 64, 0.0, 1)
    validation = subject._seed(1280000, "validation", "family", 16, 64, 0.0, 1)
    assert first == second
    assert first != validation
    assert first >> 64 == 1270000
    assert validation >> 64 == 1280000


def test_factor_selection_uses_only_coverage_cells() -> None:
    config = {
        "candidate_calibration_factors": [1.0, 1.5],
        "margin": 0.1,
        "selection_gates": {
            "calibration_min_cell_coverage": 0.95,
            "calibration_min_cell_wilson_lcb": 0.0,
        },
    }
    records = [
        {
            "split": "calibration",
            "family": "f",
            "dimension": 4,
            "cluster_count": 32,
            "true_distance": 0.1,
            "estimate": 0.04,
            "raw_radius": 0.04,
            "power_primary": True,
        }
        for _ in range(20)
    ]
    selected, diagnostics = subject.select_factor(config, records)
    assert selected == 1.5
    assert diagnostics[0]["passed"] is False
    assert diagnostics[1]["passed"] is True


def test_cluster_count_selection_requires_null_local_boundary_and_outside() -> None:
    config = {
        "cluster_counts_per_state": [64, 128],
        "true_trace_distances": [0.0, 0.05, 0.1, 0.12],
        "selection_gates": {
            "null_min_equivalence_rate": 0.8,
            "local_005_min_equivalence_rate": 0.65,
            "boundary_max_equivalence_rate": 0.1,
            "outside_max_equivalence_rate": 0.05,
        },
    }
    rows = []
    for count in (64, 128):
        rates = (
            {0.0: 0.7, 0.05: 0.6, 0.1: 0.05, 0.12: 0.0}
            if count == 64
            else {0.0: 0.9, 0.05: 0.7, 0.1: 0.05, 0.12: 0.0}
        )
        for truth, rate in rates.items():
            rows.append(
                {
                    "cluster_count": count,
                    "true_distance": truth,
                    "equivalence_rate": rate,
                    "power_primary": True,
                }
            )
    selected, diagnostics = subject.select_cluster_count(config, rows)
    assert selected == 128
    assert diagnostics[0]["passed"] is False
    assert diagnostics[1]["passed"] is True
