from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from cnn_fpga.benchmark.phase9_iq_semantics_diagnostic import (
    CALIBRATION_SEEDS,
    PASS_VERDICT,
    build_report,
    main,
    write_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live_report():
    return build_report(ROOT)


def test_fresh_diagnostic_passes_all_live_gates(live_report) -> None:
    report, rows = live_report
    assert report["verdict"] == PASS_VERDICT
    assert report["gate_summary"]["passed"] == report["gate_summary"]["total"]
    assert report["old_formal_cell_data_accessed"] is False
    assert report["margin_or_threshold_selected_from_old_outcome"] is False
    assert len(rows) == 144
    assert report["reference_step_accounting"]["observed"] == 144


def test_diagnosis_keeps_cutoff_failure_separate(live_report) -> None:
    report, _ = live_report
    diagnosis = report["historical_diagnosis"]
    assert diagnosis["iq_failures_explained_by_two_reused_rng_blocks"] == 198
    assert diagnosis["reset_coupled_failures"] == 9
    assert diagnosis["mixed_unit_fault_composite_failures"] == 8
    assert diagnosis["independent_cutoff_survival_failures"] == 2
    assert (
        diagnosis["iq_failures_explained_by_two_reused_rng_blocks"]
        + diagnosis["reset_coupled_failures"]
        + diagnosis["mixed_unit_fault_composite_failures"]
        + diagnosis["independent_cutoff_survival_failures"]
        == diagnosis["old_failed_gate_count"]
    )


def test_seed_namespaces_are_exact_and_disjoint() -> None:
    a = CALIBRATION_SEEDS["backend_a"]
    b = CALIBRATION_SEEDS["backend_b"]
    assert a == {"start": 1_030_000, "count": 512}
    assert b == {"start": 1_031_000, "count": 512}
    assert a["start"] + a["count"] <= b["start"]


def test_reference_errors_are_not_mean_only(live_report) -> None:
    report, rows = live_report
    accounting = report["reference_step_accounting"]
    assert accounting["backend_a_max_log_evidence_abs_error"] <= 2e-11
    assert accounting["backend_b_max_log_evidence_abs_error"] <= 2e-11
    assert accounting["backend_a_max_posterior_l1_error"] <= 2e-11
    assert accounting["backend_b_max_posterior_l1_error"] <= 2e-11
    assert all(float(row["integrated_mean_abs_error"]) <= 2e-12 for row in rows)


def test_rng_checks_cover_both_independent_implementations(live_report) -> None:
    report, _ = live_report
    assert set(report["rng_calibration"]) == {"backend_a", "backend_b"}
    for summary in report["rng_calibration"].values():
        assert summary["draw_pairs"] == 4096
        assert abs(summary["mean_i"]) <= 0.06
        assert abs(summary["mean_q"]) <= 0.06
        assert abs(summary["variance_i"] - 1.0) <= 0.10
        assert abs(summary["variance_q"] - 1.0) <= 0.10
        assert abs(summary["correlation_iq"]) <= 0.06


def test_diagnostic_has_no_historical_cell_artifact_literal() -> None:
    import cnn_fpga.benchmark.phase9_iq_semantics_diagnostic as diagnostic

    source = inspect.getsource(diagnostic)
    prohibited = (
        "t9_2_4_dual_backend_" + "cell_ledger.csv",
        "t9_2_4_dual_backend_" + "qualification_source_data.csv",
        "t9_2_4_dual_backend_" + "state_archive.npz",
    )
    assert not any(value in source for value in prohibited)


def test_analysis_hash_is_content_sensitive(live_report) -> None:
    report, _ = live_report
    forged = copy.deepcopy(report)
    forged["gates"]["G02_third_reference_independent"] = False
    assert forged["analysis_sha256"] == report["analysis_sha256"]
    # The retained hash cannot authenticate the mutated payload.
    from cnn_fpga.benchmark.phase9_iq_semantics_diagnostic import _sha

    assert _sha({k: v for k, v in forged.items() if k != "analysis_sha256"}) != report[
        "analysis_sha256"
    ]


def test_artifact_writer_is_complete(tmp_path: Path, monkeypatch) -> None:
    # The writer must bind live repository inputs; redirect only destinations.
    report, rows = build_report(ROOT)
    assert report["verdict"] == PASS_VERDICT
    from cnn_fpga.benchmark import phase9_iq_semantics_diagnostic as module

    monkeypatch.setattr(module, "build_report", lambda root=None: (report, rows))
    written = write_artifacts(tmp_path)
    assert written == report
    assert (tmp_path / "docs/t_risk_20260726_01_iq_semantics_diagnostic.json").is_file()
    csv_path = tmp_path / "docs/t_risk_20260726_01_iq_semantics_source_data.csv"
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 145


def test_cli_override_is_fail_closed() -> None:
    with pytest.raises(SystemExit, match="no CLI overrides"):
        main(["--seed", "1"])
