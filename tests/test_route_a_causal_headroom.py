from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import route_a_causal_headroom as audit
from cnn_fpga.benchmark.unified_comparator_runner import materialize_qualification_trace


def test_development_manifest_is_disjoint_off_grid_and_frozen() -> None:
    spec = audit.DevelopmentSpec()
    cells = audit.development_cells(spec)
    assert len(cells) == 31
    assert len({row["cell_id"] for row in cells}) == 31
    assert {row["split_id"] for row in cells} == {audit.DEVELOPMENT_SPLIT_ID}
    assert len(audit.MIXTURE_CANDIDATES) == 36
    assert tuple(row[0] for row in audit.MIXTURE_CANDIDATES[:5]) == tuple(
        f"one_hot:{method}" for method in audit.EXPERTS
    )
    old_seeds = {
        seed
        for split in audit.split_specs()
        for seed in split.seeds
    }
    assert not old_seeds.intersection(spec.seeds)


def test_observed_selector_features_do_not_accept_metadata_or_truth() -> None:
    calibration = np.asarray(
        materialize_qualification_trace()[0].calibration_residuals,
        dtype=np.float64,
    )
    static, predictors = audit._new_predictors(calibration)
    predictions = {
        "static_joint_map": static,
        **{method: predictor.prediction() for method, predictor in predictors.items()},
    }
    prior_disagreement = np.linspace(0.0, 0.2, 10)
    original = audit._observed_history_features(predictions, prior_disagreement)
    future = calibration[-audit.PARAMETER_WINDOW_DECISIONS :].copy()
    # A future residual array is not an argument to the sufficient-state
    # extractor and therefore cannot change the already-emitted feature row.
    future[:, 0] += np.linspace(0.0, 1.0, len(future))
    repeated = audit._observed_history_features(predictions, prior_disagreement)
    assert original.shape == (24,)
    assert np.array_equal(original, repeated)
    assert not np.array_equal(calibration[-audit.PARAMETER_WINDOW_DECISIONS :], future)


def test_development_cell_replay_has_true_decision_oracles_and_fail_closed_cache(
    tmp_path: Path,
) -> None:
    cell = deepcopy(audit.development_cells()[0])
    cell["scored_windows"] = 4
    cell["nominal_preamble_windows"] = 2
    calibration = np.asarray(
        materialize_qualification_trace()[0].calibration_residuals,
        dtype=np.float64,
    )
    row, hit = audit._replay_development_cell(
        cell,
        audit.DEVELOPMENT_SEEDS[0],
        calibration,
        "a" * 64,
        tmp_path,
    )
    assert hit is False
    audit._validate_development_cache(row, row["cache_context"])
    hard = np.asarray(row["hard_decision_oracle_errors_by_period"])
    expanded = np.asarray(row["expanded_action_oracle_errors_by_period"])
    period = np.min(np.asarray(row["expert_errors"]), axis=1)
    assert np.all(expanded <= hard)
    assert np.all(hard <= period)
    mutated = deepcopy(row)
    mutated["cache_context"]["seed"] += 1
    with pytest.raises(ValueError, match="provenance"):
        audit._validate_development_cache(mutated, row["cache_context"])


def test_nested_selector_is_outer_heldout_and_regrets_close() -> None:
    rng = np.random.default_rng(20260720)
    periods_per_seed = 12
    seeds = np.repeat(np.asarray(audit.DEVELOPMENT_SEEDS), periods_per_seed)
    count = len(seeds)
    features = rng.normal(size=(count, 24))
    expert_errors = rng.integers(2, 18, size=(count, len(audit.EXPERTS)))
    # Create learnable but imperfect structure without exposing truth/family.
    preferred = (features[:, 0] > 0).astype(int)
    expert_errors[np.arange(count), preferred] = rng.integers(0, 4, size=count)
    candidate_errors = rng.integers(2, 18, size=(count, len(audit.MIXTURE_CANDIDATES)))
    candidate_errors[:, : len(audit.EXPERTS)] = expert_errors
    hard_oracle = np.maximum(0, np.min(expert_errors, axis=1) - 1)
    expanded_oracle = np.maximum(0, hard_oracle - 1)
    data = {
        "features": features,
        "expert_errors": expert_errors,
        "candidate_errors": candidate_errors,
        "hard_decision_oracle_errors": hard_oracle,
        "action_oracle_errors": expanded_oracle,
        "period_decisions": np.full(count, 512, dtype=np.int64),
        "seed": seeds,
        "family": np.full(count, "not_consumed"),
        "cell_id": np.full(count, "not_consumed"),
    }
    report = audit.nested_selector_audit(data)
    assert len(report["folds"]) == len(audit.DEVELOPMENT_SEEDS)
    assert all(row["training_seed_count"] == 5 for row in report["folds"])
    regret = report["regret_decomposition"]
    assert np.isclose(
        regret["selection_regret_ler"]
        + regret["estimation_regret_ler"]
        + regret["action_space_regret_ler"],
        regret["identity_total_ler"],
        atol=1e-15,
        rtol=0,
    )


def test_one_formal_trajectory_rebinds_parent_exactly(tmp_path: Path) -> None:
    parents, bindings = audit._formal_parent_rows()
    cell = next(
        row
        for row in audit.scenario_cells()
        if row["split_id"] == audit.FORMAL_SPLIT_ID
    )
    seed = next(
        spec.seeds[0]
        for spec in audit.split_specs()
        if spec.split_id == audit.FORMAL_SPLIT_ID
    )
    key = (str(cell["family"]), str(cell["cell_id"]), int(seed))
    calibration = np.asarray(
        materialize_qualification_trace()[0].calibration_residuals,
        dtype=np.float64,
    )
    row, hit = audit._replay_formal_cell(
        cell, seed, parents[key], calibration, bindings, tmp_path
    )
    assert hit is False
    assert row["parent_exact_match"] is True
    assert int(sum(row["period_decisions"])) == row["scored_decisions"]
    audit._validate_formal_cell_cache(row, row["cache_context"])
