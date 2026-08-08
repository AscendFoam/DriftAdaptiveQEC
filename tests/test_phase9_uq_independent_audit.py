from __future__ import annotations

from copy import deepcopy

import pytest

from cnn_fpga.benchmark import phase9_uq_independent_audit as subject


def _row(**overrides):
    row = {
        "split": "validation",
        "family": "f",
        "dimension": 16,
        "cluster_count": 384,
        "true_distance": 0.0,
        "trial": 0,
        "trial_seed": (1350000 << 64) | 1,
        "multiplier_seed": (1360000 << 64) | 1,
        "estimate": 0.01,
        "raw_radius": 0.02,
        "factor": 1.0,
        "calibrated_upper_bound": 0.03,
        "covered": True,
        "equivalent": True,
        "power_primary": True,
    }
    row.update(overrides)
    return row


def test_source_arithmetic_and_duplicate_key_fail_closed() -> None:
    row = _row()
    cells = subject._cell_rows([row], 1.0)
    assert cells[0]["coverage_rate"] == 1.0
    with pytest.raises(ValueError, match="duplicate"):
        subject._cell_rows([row, deepcopy(row)], 1.0)
    corrupted = _row(calibrated_upper_bound=0.031)
    with pytest.raises(ValueError, match="arithmetic"):
        subject._cell_rows([corrupted], 1.0)


def test_seed_firewall_checks_high_namespace_and_uniqueness() -> None:
    rows = [_row()]
    assert subject._seed_firewall(
        rows, {"validation": 1350000}, 1360000
    )
    rows.append(_row(trial=1, trial_seed=rows[0]["trial_seed"]))
    assert not subject._seed_firewall(
        rows, {"validation": 1350000}, 1360000
    )


def test_split_specific_coverage_thresholds() -> None:
    gates = {
        "calibration_min_cell_coverage": 0.95,
        "calibration_min_cell_wilson_lcb": 0.9,
        "validation_min_cell_coverage": 0.94,
        "validation_min_cell_wilson_lcb": 0.88,
    }
    cells = [
        {
            "split": "calibration",
            "coverage_rate": 0.945,
            "coverage_wilson_lcb": 0.89,
        }
    ]
    assert subject._coverage_passed(cells, gates, "calibration") is False
    cells[0]["split"] = "validation"
    assert subject._coverage_passed(cells, gates, "validation") is True


def test_claim_boundary_requires_only_scope_marker_nonnull() -> None:
    assert subject._claims_scoped_null(
        {"calibration_only": True, "ler": None}, "calibration_only"
    )
    assert not subject._claims_scoped_null(
        {"calibration_only": True, "ler": False}, "calibration_only"
    )
