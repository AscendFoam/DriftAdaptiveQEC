from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from cnn_fpga.benchmark.route_a_integrated_rtl_qualification import (
    FAMILY_CYCLES,
    ROOT,
    TRACE_STRUCT,
    evaluate_gates,
)
from cnn_fpga.runtime.fast_production_core_reference import crc16_int_little_endian
from cnn_fpga.runtime.route_a_fixed_policy_reference import (
    ACTION_INTEGRITY_ROLLBACK,
    ACTION_OPEN,
    ACTION_TAIL_EWMA,
    ACTION_UNCERTAIN_EWMA,
    RouteAFixedPolicyReference,
    RouteAPolicyInputs,
)


REPORT = ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json"
SOURCE_DATA = ROOT / "docs/t6_7_3_route_a_integrated_rtl_source_data.csv"


def _step(model: RouteAFixedPolicyReference, inputs: RouteAPolicyInputs):
    return model.step(
        inputs,
        sample_valid=1,
        safe_boundary=1,
        active_bank=0,
        active_version=0,
        core_output_word=1,
    )


def test_integer_threshold_sides_are_exact() -> None:
    below = RouteAFixedPolicyReference()
    row = _step(
        below,
        RouteAPolicyInputs(
            posterior_valid=1,
            p_normal=26,
            p_smooth=0,
            p_calibration=229,
            p_burst=0,
        ),
    )
    assert row.action == ACTION_UNCERTAIN_EWMA
    assert row.tail_latched == 0

    at = RouteAFixedPolicyReference()
    row = _step(
        at,
        RouteAPolicyInputs(
            posterior_valid=1,
            p_normal=25,
            p_smooth=0,
            p_calibration=230,
            p_burst=0,
        ),
    )
    assert row.action == ACTION_TAIL_EWMA
    assert row.tail_latched == 0


def test_two_enter_and_eight_recovery_updates_drive_hysteresis() -> None:
    model = RouteAFixedPolicyReference()
    tail = RouteAPolicyInputs(
        posterior_valid=1,
        p_normal=10,
        p_smooth=5,
        p_calibration=120,
        p_burst=120,
    )
    assert _step(model, tail).tail_latched == 0
    assert _step(model, tail).tail_latched == 1
    healthy = RouteAPolicyInputs(
        posterior_valid=1,
        p_normal=235,
        p_smooth=20,
        p_calibration=0,
        p_burst=0,
    )
    for _ in range(7):
        assert _step(model, healthy).tail_latched == 1
    recovered = _step(model, healthy)
    assert recovered.tail_latched == 0
    assert recovered.action == ACTION_OPEN


def test_posterior_sum_and_version_faults_fail_closed_with_distinct_reason() -> None:
    sum_model = RouteAFixedPolicyReference()
    sum_fault = _step(
        sum_model,
        RouteAPolicyInputs(
            posterior_valid=1,
            p_normal=100,
            p_smooth=100,
            p_calibration=30,
            p_burst=24,
        ),
    )
    assert (sum_fault.action, sum_fault.reason) == (ACTION_INTEGRITY_ROLLBACK, 7)

    version_model = RouteAFixedPolicyReference()
    version_fault = _step(version_model, RouteAPolicyInputs(version_fault=1))
    assert (version_fault.action, version_fault.reason) == (ACTION_INTEGRITY_ROLLBACK, 8)


def test_window_router_creates_only_monotonic_auto_commit() -> None:
    model = RouteAFixedPolicyReference()
    route = _step(
        model,
        RouteAPolicyInputs(
            posterior_valid=1,
            p_normal=20,
            p_smooth=225,
            p_calibration=5,
            p_burst=5,
            router_boundary=1,
            window_prequential_win=1,
        ),
    )
    assert route.selected_bank == 1
    assert route.commit_pending == 1
    assert model.peek_auto_commit(safe_boundary=0, active_bank=0, active_version=9) == (0, 1, 10)
    assert model.peek_auto_commit(safe_boundary=1, active_bank=0, active_version=9) == (1, 1, 10)
    assert model.peek_auto_commit(safe_boundary=1, active_bank=0, active_version=0xFFFF)[0] == 0


def test_route_words_are_crc_protected_and_pipeline_aligned() -> None:
    model = RouteAFixedPolicyReference()
    first = RouteAPolicyInputs(
        posterior_valid=1,
        p_normal=10,
        p_smooth=5,
        p_calibration=120,
        p_burst=120,
    )
    rows = [_step(model, first)]
    rows.extend(_step(model, RouteAPolicyInputs()) for _ in range(6))
    aligned = rows[-1]
    assert aligned.action_word & 1
    assert ((aligned.action_word >> 1) & 7) == ACTION_TAIL_EWMA
    for word, bits, byte_count in (
        (aligned.action_word, 64, 8),
        (aligned.state_word, 80, 10),
        (aligned.version_word, 48, 6),
    ):
        payload = word & ((1 << bits) - 1)
        assert (word >> bits) == crc16_int_little_endian(payload, byte_count)


def test_formal_report_recomputes_all_gates_and_source_hashes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["cycles_per_family"] == FAMILY_CYCLES
    assert report["trace"]["row_bytes"] == TRACE_STRUCT.size == 131
    assert report["trace"]["rows"] == 1_000_000
    assert report["verdict"] == "PASS_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION"
    assert all(evaluate_gates(report).values())
    for relative, expected in report["source_hashes"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_source_data_indexes_all_ten_full_families() -> None:
    with SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert all(int(row["cycles"]) == 100_000 for row in rows)
    assert all(int(row["rtl_mismatches"]) == 0 for row in rows)
    assert sum(int(row["silent_overflow"]) for row in rows) == 0
