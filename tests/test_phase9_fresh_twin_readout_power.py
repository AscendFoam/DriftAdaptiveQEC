from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from cnn_fpga.benchmark.phase9_fresh_twin_readout_power import (
    PASS_VERDICT,
    PRIORS,
    build_report,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live_readout_power():
    return build_report(ROOT)


def test_empirical_readout_power_passes_all_gates(live_readout_power) -> None:
    report, rows = live_readout_power
    assert report["verdict"] == PASS_VERDICT
    assert report["gate_summary"] == {"passed": 19, "total": 19}
    assert report["selected_round_sample_count"] in {128, 256, 512, 768}
    assert len(rows) == 12
    assert report["fresh_stream_draws"] == 6 * 768 * len(PRIORS)


def test_same_backend_and_ab_cases_are_all_empirical(live_readout_power) -> None:
    _, rows = live_readout_power
    assert {row["case"] for row in rows} == {
        "backend_a_a1_a2",
        "backend_b_b1_b2",
        "ab_pilot",
    }
    assert all(row["trials"] == 2000 for row in rows)
    assert all(row["metric_count"] == len(PRIORS) * 5 for row in rows)


def test_common_heldout_score_does_not_use_independent_raw_score(
    live_readout_power,
) -> None:
    report, _ = live_readout_power
    score = report["proper_score_and_llr"]
    assert score["evaluation"] == "common-heldout paired semantic identity"
    assert score["max_score_error"] == 0.0
    assert score["max_llr_error"] == 0.0
    assert score["independent_raw_log_evidence_primary"] is False
    assert max(report["raw_sampler_diagnostic_not_primary"].values()) <= 1.0


def test_seed_cluster_dependence_is_preserved(live_readout_power) -> None:
    report, _ = live_readout_power
    assert report["cluster_unit"] == "seed_position shared across all prior archetypes"
    assert report["formal_seed_pool_accessed"] is False
    assert report["historical_formal_cell_data_accessed"] is False


def test_no_old_cell_artifact_literal_or_formal_rng_use() -> None:
    import cnn_fpga.benchmark.phase9_fresh_twin_readout_power as module

    source = inspect.getsource(module)
    prohibited = (
        "t9_2_4_dual_backend_" + "cell_ledger.csv",
        "t9_2_4_dual_backend_" + "qualification_source_data.csv",
        "t9_2_4_dual_backend_" + "state_archive.npz",
    )
    assert not any(value in source for value in prohibited)
    assert "formal_round_backend_a" not in source
    assert "formal_round_backend_b" not in source


def test_repeat_is_bit_deterministic(live_readout_power) -> None:
    first, first_rows = live_readout_power
    second, second_rows = build_report(ROOT)
    assert first == second
    assert first_rows == second_rows


def test_cli_overrides_fail_closed() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--count", "8"])
    assert raised.value.code == 2
