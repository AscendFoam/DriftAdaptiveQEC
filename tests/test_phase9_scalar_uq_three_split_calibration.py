from __future__ import annotations

import json
import ast
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from cnn_fpga.benchmark import phase9_scalar_uq_three_split_calibration as subject
from cnn_fpga.benchmark import phase9_scalar_uq_three_split_verifier as verifier


ROOT = Path(__file__).resolve().parents[1]


def _raw_rows(
    cell: subject.Cell, count: int, *, split: str = "selection_a"
) -> list[dict[str, object]]:
    rows = []
    for trial in range(count):
        if cell.effect_ratio < 1.0:
            estimate = cell.effect_ratio * cell.margin
        elif cell.effect_ratio == 1.0:
            estimate = 1.05 * cell.margin
        else:
            estimate = cell.effect_ratio * cell.margin
        rows.append(
            {
                "split": split,
                "cell_id": cell.cell_id,
                "family": cell.family,
                "margin": cell.margin,
                "cluster_count": cell.cluster_count,
                "effect_ratio": cell.effect_ratio,
                "true_difference": cell.effect_ratio * cell.margin,
                "trial_index": trial,
                "trial_seed": 1000 + trial,
                "multiplier_seed": 2000 + trial,
                "estimate": estimate,
                "raw_radius": 0.0,
            }
        )
    return rows


def _passing_summaries(config) -> list[dict[str, object]]:
    rows = []
    for cell in subject._cells(config):
        if cell.effect_ratio < 1.0:
            eq_lcb, eq_ucb = 1.0, 1.0
        else:
            eq_lcb, eq_ucb = 0.0, 0.0
        rows.append(
            {
                "cell_id": cell.cell_id,
                "family": cell.family,
                "margin": cell.margin,
                "cluster_count": cell.cluster_count,
                "effect_ratio": cell.effect_ratio,
                "power_primary": config["families"][cell.family]["power_primary"],
                "coverage_successes": config["trial_count_per_cell"],
                "coverage_rate": 1.0,
                "coverage_wilson_lcb": 1.0,
                "coverage_wilson_ucb": 1.0,
                "equivalence_successes": (
                    config["trial_count_per_cell"] if cell.effect_ratio < 1.0 else 0
                ),
                "equivalence_rate": 1.0 if cell.effect_ratio < 1.0 else 0.0,
                "equivalence_wilson_lcb": eq_lcb,
                "equivalence_wilson_ucb": eq_ucb,
            }
        )
    return rows


def _selection_payload(passing_factors: set[float], split: str):
    return {
        "split": split,
        "evaluated_factors": list(subject.FACTOR_GRID),
        "factor_gates": {
            f"{factor:.1f}": {"global_pass": factor in passing_factors}
            for factor in subject.FACTOR_GRID
        },
    }


def test_live_contract_has_three_disjoint_splits_and_immutable_parent() -> None:
    config = subject.load_config(ROOT)
    assert len(subject._cells(config)) == 192
    assert config["trial_count_per_cell"] == 2048
    assert config["factor_grid"] == subject.FACTOR_GRID
    assert config["claim_boundary"] == subject.CLAIM_BOUNDARY
    seeds = {
        value[key]
        for value in config["splits"].values()
        for key in ("trial_seed_base", "multiplier_seed_base")
    }
    seeds |= {
        config["resource_preflight"]["trial_seed_base"],
        config["resource_preflight"]["multiplier_seed_base"],
    }
    assert len(seeds) == 8
    parent = config["diagnostic_only_parent"]
    assert parent["report"]["sha256"] == "d8930ca946a8e0cf83c37af905a7dc85e60ae5e37462a77cb32f8dce880ffc60"
    assert parent["source_data"]["sha256"] == "ebf5d6bc21e6e7f38018fc2d1402372943937afb13db9606fb91fd2087654689"
    failure = config["infrastructure_failure_parent"]
    assert (
        failure["terminal_state"]
        == "FAILED_INFRASTRUCTURE_SEED_COLLISION_BEFORE_SELECTION_SEAL"
    )
    assert (
        failure["v1_outcomes_used_as_v2_selection_or_confirmation_evidence"]
        is False
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("factor_grid",), [1.0]),
        (("trial_count_per_cell",), 288),
        (("splits", "confirmation", "trial_seed_base"), 2000000),
        (("simultaneous_wilson", "comparisons_per_split"), 192),
        (("gates", "minimum_cell_coverage_wilson_lcb"), 0.8),
        (("design_outcomes_accessed",), True),
        (("claim_boundary", "external_sota"), True),
    ],
)
def test_preregistered_contract_mutations_fail_closed(
    tmp_path, monkeypatch, path, value
) -> None:
    config = json.loads((ROOT / subject.CONFIG_PATH).read_text(encoding="utf-8"))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    candidate = tmp_path / "mutated.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(subject, "CONFIG_PATH", "mutated.json")
    monkeypatch.setattr(subject, "_validate_parent_diagnostic", lambda *_: None)
    with pytest.raises(ValueError):
        subject.load_config(tmp_path)


def test_2048_trial_simultaneous_wilson_gate_is_mathematically_feasible() -> None:
    config = subject.load_config(ROOT)
    lcb, _ = subject._wilson(round(0.95 * 2048), 2048, config)
    assert lcb > subject.GATES["minimum_cell_coverage_wilson_lcb"]
    old_lcb, _ = subject._wilson(round(0.95 * 288), 288, config)
    assert old_lcb < subject.GATES["minimum_cell_coverage_wilson_lcb"]


def test_v2_seed_map_is_injective_and_regresses_the_observed_v1_collision() -> None:
    config = subject.load_config(ROOT)
    trial_count = config["trial_count_per_cell"]
    intervals = sorted(
        (
            value[key],
            value[key] + config["seed_addressing"]["maximum_offset"],
        )
        for value in config["splits"].values()
        for key in ("trial_seed_base", "multiplier_seed_base")
    )
    assert all(
        left[1] < right[0]
        for left, right in zip(intervals, intervals[1:])
    )
    seeds = {
        subject._split_seed(2_000_000, cell, trial, trial_count)
        for cell in range(192)
        for trial in range(trial_count)
    }
    assert len(seeds) == 192 * trial_count

    def old_address(base: int, *parts: object) -> int:
        digest = sha256("|".join(map(str, parts)).encode()).digest()
        return base + int.from_bytes(digest[:8], "big") % 1_000_000_000

    cell_id = "rare_heavy_tail__m0p005000__n384__r0p500"
    assert old_address(1_630_000, "selection_b", cell_id, 358) == old_address(
        1_630_000, "selection_b", cell_id, 908
    )
    assert subject._split_seed(4_000_000, 105, 358, trial_count) != (
        subject._split_seed(4_000_000, 105, 908, trial_count)
    )


def test_factor_gate_covers_all_192_cells_and_64_primary_power_strata() -> None:
    config = subject.load_config(ROOT)
    gate = subject._factor_gate(1.2, _passing_summaries(config), config)
    assert gate["global_pass"] is True
    assert gate["coverage_failed_cells"] == []
    assert sum(row["stratum_count"] for row in gate["power_ledger"]) == 64
    assert all(row["global_iut_pass"] for row in gate["power_ledger"])


def test_selection_requires_both_folds_and_returns_smallest_factor() -> None:
    a = _selection_payload({1.1, 1.2, 1.3, 1.4, 1.5}, "selection_a")
    b = _selection_payload({1.2, 1.3, 1.4, 1.5}, "selection_b")
    assert subject._select_factor(a, b) == 1.2
    b = _selection_payload(set(), "selection_b")
    assert subject._select_factor(a, b) is None


def test_raw_row_semantics_indices_and_seed_uniqueness_are_enforced() -> None:
    config = subject.load_config(ROOT)
    config = json.loads(json.dumps(config))
    config["trial_count_per_cell"] = 3
    cell = subject.Cell("rare_heavy_tail", 0.1, 384, 0.5)
    rows = _raw_rows(cell, 3)
    summary = subject._evaluate_cell(cell, rows, 1.2, config)
    assert summary["coverage_successes"] == 3
    rows[0]["true_difference"] = 0.0
    with pytest.raises(RuntimeError, match="semantic"):
        subject._evaluate_cell(cell, rows, 1.2, config)


def test_factor_inflation_is_monotone_for_coverage_and_equivalence() -> None:
    cell = subject.Cell("rare_heavy_tail", 0.08, 384, 1.0)
    row = subject._one_trial_raw(
        "selection_a",
        cell,
        subject.FAMILIES[cell.family],
        trial_index=0,
        trial_seed=123,
        multiplier_seed=456,
        confidence=0.95,
        replicates=199,
    )
    low = float(row["estimate"]) + 1.0 * float(row["raw_radius"])
    high = float(row["estimate"]) + 1.5 * float(row["raw_radius"])
    assert high >= low
    assert np.isfinite(high)


def test_single_blas_thread_policy_preserves_raw_trial() -> None:
    cell = subject.Cell("gaussian_balanced", 0.1, 384, 0.5)
    kwargs = dict(
        split="selection_a",
        cell=cell,
        family=subject.FAMILIES[cell.family],
        trial_index=0,
        trial_seed=123,
        multiplier_seed=456,
        confidence=0.95,
        replicates=199,
    )
    with threadpool_limits(limits=1, user_api="blas"):
        one = subject._one_trial_raw(**kwargs)
        libraries = [
            info for info in threadpool_info() if info.get("user_api") == "blas"
        ]
    with threadpool_limits(limits=4, user_api="blas"):
        four = subject._one_trial_raw(**kwargs)
    assert libraries
    assert all(int(info["num_threads"]) == 1 for info in libraries)
    assert one["estimate"] == pytest.approx(four["estimate"], abs=1e-14)
    assert one["raw_radius"] == pytest.approx(four["raw_radius"], abs=1e-14)


def test_confirmation_cannot_evaluate_or_reselect_multiple_factors(
    monkeypatch,
) -> None:
    config = subject.load_config(ROOT)
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda *_: {"path": "x", "bytes": 1, "sha256": "0" * 64},
    )
    identity = {
        "run_id": "run",
        "release_commit": "a" * 40,
        "analysis_sha256": "b" * 64,
    }
    resource = {"analysis_sha256": "c" * 64}
    receipt = {
        "selected_factor": 1.2,
        "analysis_sha256": "d" * 64,
        "selection_a": {},
        "selection_b": {},
    }
    confirmation = {
        "evaluated_factors": [1.2, 1.3],
        "factor_gates": {"1.2": {"global_pass": True}},
    }
    with pytest.raises(RuntimeError, match="reselection"):
        subject._build_final_report(
            ROOT, config, identity, resource, receipt, confirmation
        )


def test_pass_report_keeps_all_external_claims_null(monkeypatch) -> None:
    config = subject.load_config(ROOT)
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda *_: {"path": "x", "bytes": 1, "sha256": "0" * 64},
    )
    identity = {
        "run_id": "run",
        "release_commit": "a" * 40,
        "analysis_sha256": "b" * 64,
    }
    receipt = {
        "selected_factor": 1.2,
        "analysis_sha256": "d" * 64,
        "selection_a": {
            "source_data_binding": {
                "path": "selection_a.csv",
                "bytes": 1,
                "sha256": "1" * 64,
            }
        },
        "selection_b": {
            "source_data_binding": {
                "path": "selection_b.csv",
                "bytes": 1,
                "sha256": "2" * 64,
            }
        },
    }
    confirmation = {
        "evaluated_factors": [1.2],
        "factor_gates": {"1.2": {"global_pass": True}},
        "source_data_binding": {
            "path": "confirmation.csv",
            "bytes": 1,
            "sha256": "3" * 64,
        },
    }
    report = subject._build_final_report(
        ROOT,
        config,
        identity,
        {"analysis_sha256": "c" * 64},
        receipt,
        confirmation,
    )
    assert report["verdict"] == subject.PASS_VERDICT
    assert report["diagnostic_parent_used_as_selection_or_confirmation_evidence"] is False
    assert report["repair_contract_frozen_after_diagnostic"] is True
    assert (
        report["v1_failure_used_as_v2_selection_or_confirmation_evidence"]
        is False
    )
    assert report["claim_state"]["scalar_uq_calibration_only"] is True
    assert all(
        value is None
        for key, value in report["claim_state"].items()
        if key != "scalar_uq_calibration_only"
    )


def test_owner_lock_is_exclusive_and_removed(tmp_path) -> None:
    config = subject.load_config(ROOT)
    config = json.loads(json.dumps(config))
    config["artifact_paths"]["owner_lock"] = "run/owner.lock"
    with subject._owner_lock(tmp_path, config):
        with pytest.raises(FileExistsError):
            with subject._owner_lock(tmp_path, config):
                pass
    assert not (tmp_path / "run/owner.lock").exists()


def test_real_process_pool_split_smoke_is_lossless_and_ordered(tmp_path) -> None:
    config = subject.load_config(ROOT)
    config = json.loads(json.dumps(config))
    config["trial_count_per_cell"] = 2
    config["max_workers"] = 2
    config["artifact_paths"]["selection_a_source_data"] = "selection_a.csv"
    result = subject._write_split(
        tmp_path, config, "selection_a", [1.0]
    )
    assert result["raw_trial_count"] == 384
    assert result["cell_count"] == 192
    assert result["all_workers_single_blas_thread"] is True
    assert result["evaluated_factors"] == [1.0]
    path = tmp_path / "selection_a.csv"
    assert sum(1 for _ in path.open("r", encoding="utf-8")) == 385
    lines = path.read_text(encoding="utf-8").splitlines()
    first_cell = subject._cells(config)[0].cell_id
    second_cell = subject._cells(config)[1].cell_id
    assert first_cell in lines[1] and first_cell in lines[2]
    assert second_cell in lines[3] and second_cell in lines[4]


def test_help_is_zero_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(subject, "_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as exc:
        subject.main(["--help"])
    assert exc.value.code == 0
    assert not list(tmp_path.rglob("*"))


def test_unknown_argument_is_rejected_without_writes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(subject, "_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as exc:
        subject.main(["--factor", "1.5"])
    assert exc.value.code == 2
    assert not list(tmp_path.rglob("*"))


def test_independent_verifier_does_not_import_production_runner() -> None:
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        "phase9_scalar_uq_three_split_calibration" in name for name in imports
    )


def test_independent_verifier_seed_and_wilson_recompute_match() -> None:
    config = subject.load_config(ROOT)
    assert verifier._split_seed(
        2_000_000, 17, 23, 2048
    ) == subject._split_seed(2_000_000, 17, 23, 2048)
    assert verifier._wilson(1946, 2048) == pytest.approx(
        subject._wilson(1946, 2048, config), abs=1e-15
    )


def test_independent_verifier_factor_gate_matches_production() -> None:
    config = subject.load_config(ROOT)
    summaries = _passing_summaries(config)
    assert verifier._factor_gate(1.2, summaries) == subject._factor_gate(
        1.2, summaries, config
    )


def test_independent_self_hash_rejects_mutation() -> None:
    payload = {"schema_version": "x", "value": 1}
    payload["analysis_sha256"] = verifier._sha(payload)
    verifier._self_hash(payload, "fixture")
    payload["value"] = 2
    with pytest.raises(RuntimeError, match="self-hash"):
        verifier._self_hash(payload, "fixture")


def test_independent_verifier_help_is_zero_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(verifier, "_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as exc:
        verifier.main(["--help"])
    assert exc.value.code == 0
    assert not list(tmp_path.rglob("*"))
