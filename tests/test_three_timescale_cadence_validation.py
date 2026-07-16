from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from cnn_fpga.benchmark.three_timescale_cadence_validation import (
    DEFAULT_JSON,
    DEFAULT_SWEEP_CSV,
    DEFAULT_TRACE_CSV,
    ROOT,
    SCHEMA_VERSION,
    run_validation,
)


def test_validation_passes_all_gates_and_writes_source_data(tmp_path: Path) -> None:
    json_path = tmp_path / "validation.json"
    sweep_path = tmp_path / "sweep.csv"
    trace_path = tmp_path / "trace.csv"
    result = run_validation(
        json_path=json_path,
        sweep_csv_path=sweep_path,
        trace_csv_path=trace_path,
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert len(result["gates"]) == 14
    assert all(gate["passed"] for gate in result["gates"])
    assert result["source_data"]["phase_sweep_rows"] == 8000
    assert json_path.exists() and sweep_path.exists() and trace_path.exists()


def test_phase_source_data_has_both_policies_and_exact_boundaries(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.csv"
    run_validation(
        json_path=tmp_path / "result.json",
        sweep_csv_path=sweep_path,
        trace_csv_path=tmp_path / "trace.csv",
    )
    with sweep_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    by_policy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_policy.setdefault(row["evidence_policy"], []).append(row)
    assert set(by_policy) == {
        "first_influenced_window",
        "first_full_post_change_window",
    }
    assert {len(rows) for rows in by_policy.values()} == {4000}
    assert min(int(row["total_lag_cycles"]) for row in by_policy["first_influenced_window"]) == 200
    assert max(int(row["total_lag_cycles"]) for row in by_policy["first_full_post_change_window"]) == 6246


def test_trace_connects_t42_event_path_scheduler_and_version_commit(tmp_path: Path) -> None:
    result = run_validation(
        json_path=tmp_path / "result.json",
        sweep_csv_path=tmp_path / "sweep.csv",
        trace_csv_path=tmp_path / "trace.csv",
    )
    trace = result["reference_execution_trace"]
    observed = trace["observed"]

    assert observed["event_action_epoch"] - observed["event_source_epoch"] == 1
    assert observed["event_action_mode"] == "hold"
    assert observed["precommit_fast_version"] == 0
    assert observed["first_fast_use_version"] == 1
    assert observed["commit_epoch"] == trace["expected"]["first_use_epoch"]


def test_recalibration_is_due_only_and_never_claimed_as_direct_mutation(tmp_path: Path) -> None:
    result = run_validation(
        json_path=tmp_path / "result.json",
        sweep_csv_path=tmp_path / "sweep.csv",
        trace_csv_path=tmp_path / "trace.csv",
    )
    text = result["cadence_definition"]["recalibration"]
    rows = result["recalibration_schedule_example"]

    assert "due signal" in text
    assert "never direct active-bank mutation" in text
    assert rows[-1]["kinds"] == ["end_of_run"]


def test_hil_configs_freeze_age_limit_above_two_slow_periods() -> None:
    for relative in (
        "cnn_fpga/config/hardware_hil.yaml",
        "cnn_fpga/config/hardware_emulation.yaml",
    ):
        raw = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        runtime = raw["runtime"]
        slow_cycles = int(runtime["t_slow_update_ms"] * 1000 / runtime["t_fast_us"])
        assert runtime["max_parameter_age_cycles"] >= 2 * slow_cycles


def test_production_artifacts_are_self_consistent_when_present() -> None:
    if not (DEFAULT_JSON.exists() and DEFAULT_SWEEP_CSV.exists() and DEFAULT_TRACE_CSV.exists()):
        return
    artifact = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    with DEFAULT_SWEEP_CSV.open(newline="", encoding="utf-8") as stream:
        sweep_rows = sum(1 for _ in csv.DictReader(stream))
    with DEFAULT_TRACE_CSV.open(newline="", encoding="utf-8") as stream:
        trace_rows = sum(1 for _ in csv.DictReader(stream))

    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["source_data"]["phase_sweep_rows"] == sweep_rows == 8000
    assert artifact["source_data"]["trace_rows"] == trace_rows
    assert all(gate["passed"] for gate in artifact["gates"])


def test_human_contract_and_document_maps_are_synchronized() -> None:
    human = (ROOT / "docs" / "three_timescale_cadence.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "protocol_hierarchy.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmark_readme = (ROOT / "cnn_fpga" / "benchmark" / "README.md").read_text(
        encoding="utf-8"
    )

    for fragment in (
        "first_influenced_window",
        "first_full_post_change_window",
        "8192 cycles",
        "12,000,000 cycles",
        "不是",
    ):
        assert fragment in human
    assert "3.35 Three-timescale cadence and adaptation lag" in protocol
    assert "three_timescale_cadence.md" in root_readme
    assert "three_timescale_cadence_validation.py" in benchmark_readme
