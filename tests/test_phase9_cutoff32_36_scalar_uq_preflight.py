from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from cnn_fpga.benchmark import (
    phase9_cutoff32_36_scalar_uq_preflight as subject,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rows(config, *, break_coverage: bool = False):
    rows = []
    trials = config["trial_count_per_cell"]
    for cell in subject._cells(config):
        for trial in range(trials):
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "covers_truth": not (
                        break_coverage
                        and cell == subject._cells(config)[0]
                        and trial < 32
                    ),
                    "declares_equivalent": cell.effect_ratio < 1.0,
                }
            )
    return rows


def test_live_config_freezes_all_margins_counts_and_claim_nulls() -> None:
    config = subject.load_config(ROOT)
    assert len(subject._cells(config)) == 192
    assert config["cluster_counts"] == [12, 384]
    assert config["margins"] == [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.25]
    assert config["multiplier_replicates"] == 199
    assert config["numeric_execution"]["blas_threads_per_worker"] == 1
    assert set(config["claim_boundary"].values()) == {True, None}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("margins",), [0.01]),
        (("cluster_counts",), [384]),
        (("multiplier_replicates",), 1999),
        (("families", "rare_heavy_tail", "rare_probability"), 0.01),
        (("design_outcomes_accessed",), True),
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
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(subject, "CONFIG_PATH", "mutated.json")
    with pytest.raises(ValueError):
        subject.load_config(tmp_path)


@pytest.mark.parametrize("family", list(subject.FAMILIES))
def test_scalar_generator_is_finite_deterministic_and_margin_scaled(family) -> None:
    cell = subject.Cell(family, 0.005, 384, 0.5)
    kwargs = {
        "trial_seed": 123,
        "multiplier_seed": 456,
        "confidence": 0.95,
        "replicates": 199,
        "factor": 1.0,
    }
    first = subject._one_trial(cell, subject.FAMILIES[family], **kwargs)
    second = subject._one_trial(cell, subject.FAMILIES[family], **kwargs)
    assert first == second
    assert first["true_difference"] == pytest.approx(0.0025)
    assert np.isfinite(first["upper_bound"])
    assert first["upper_bound"] >= first["estimate"] >= 0.0


def test_single_blas_thread_policy_preserves_scalar_ucb() -> None:
    cell = subject.Cell("rare_heavy_tail", 0.1, 384, 0.5)
    kwargs = {
        "trial_seed": 123,
        "multiplier_seed": 456,
        "confidence": 0.95,
        "replicates": 199,
        "factor": 1.0,
    }
    with threadpool_limits(limits=1, user_api="blas"):
        one = subject._one_trial(cell, subject.FAMILIES[cell.family], **kwargs)
        libraries = [
            info
            for info in threadpool_info()
            if info.get("user_api") == "blas"
        ]
    with threadpool_limits(limits=4, user_api="blas"):
        four = subject._one_trial(cell, subject.FAMILIES[cell.family], **kwargs)
    assert libraries
    assert all(int(info["num_threads"]) == 1 for info in libraries)
    assert one["estimate"] == pytest.approx(four["estimate"], abs=1e-14)
    assert one["raw_radius"] == pytest.approx(four["raw_radius"], abs=1e-14)
    assert one["upper_bound"] == pytest.approx(four["upper_bound"], abs=1e-14)


def test_global_iut_pass_and_claim_firewall() -> None:
    config = subject.load_config(ROOT)
    report, summaries = subject._build_report(
        ROOT,
        config,
        _synthetic_rows(config),
        {"analysis_sha256": "resource", "passed": True},
    )
    assert len(summaries) == 192
    assert report["verdict"] == subject.PASS_VERDICT
    assert report["coverage_all_passed"] is True
    assert report["power_all_passed"] is True
    assert report["qualified_claim"] is None
    assert set(report["claim_state"].values()) == {True, None}


def test_any_coverage_cell_failure_is_global_no_go() -> None:
    config = subject.load_config(ROOT)
    report, _ = subject._build_report(
        ROOT,
        config,
        _synthetic_rows(config, break_coverage=True),
        {"analysis_sha256": "resource", "passed": True},
    )
    assert report["verdict"] == subject.NO_GO_VERDICT
    assert report["coverage_all_passed"] is False
    assert report["qualified_claim"] is None


def test_missing_trial_is_rejected() -> None:
    config = subject.load_config(ROOT)
    rows = _synthetic_rows(config)
    with pytest.raises(RuntimeError, match="denominator"):
        subject._build_report(
            ROOT,
            config,
            rows[:-1],
            {"analysis_sha256": "resource", "passed": True},
        )


def test_owner_lock_is_exclusive_and_cleaned(tmp_path) -> None:
    config = subject.load_config(ROOT)
    config = json.loads(json.dumps(config))
    config["artifact_paths"]["owner_lock"] = "run/owner.lock"
    with subject._owner_lock(tmp_path, config):
        with pytest.raises(FileExistsError):
            with subject._owner_lock(tmp_path, config):
                pass
    assert not (tmp_path / "run/owner.lock").exists()


def test_help_is_zero_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(subject, "_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as exc:
        subject.main(["--help"])
    assert exc.value.code == 0
    assert not list(tmp_path.rglob("*"))
