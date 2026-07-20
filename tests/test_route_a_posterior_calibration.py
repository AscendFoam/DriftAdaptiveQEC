from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.route_a_posterior_calibration import (
    BASELINE_METHODS,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    _trace_seed,
    verify_report,
)
from cnn_fpga.benchmark.route_a_preregistration import protocol_payload, scenario_cells
from cnn_fpga.decoder.route_a_regime_posterior import (
    ROUTE_A_POSTERIOR_CLASSES,
    ObservedTailEventModel,
    RouteAPosteriorModel,
    temperature_scale,
)


def _report() -> dict[str, object]:
    return json.loads(Path(DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


def test_route_a_model_rejects_legacy_class_order_and_temperature_underflow() -> None:
    report = _report()
    payload = report["posterior_model"]
    model = RouteAPosteriorModel.from_payload(payload)
    assert model.class_order == ROUTE_A_POSTERIOR_CLASSES
    mutated = json.loads(json.dumps(payload))
    mutated["class_order"] = ["normal", "burst", "leakage", "calibration_shift"]
    with pytest.raises(ValueError, match="class order"):
        RouteAPosteriorModel.from_payload(mutated)

    posterior = np.asarray(((1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)))
    scaled = temperature_scale(posterior, 2.0)
    assert np.all(np.isfinite(scaled))
    assert np.all(scaled >= 0.0)
    assert np.allclose(np.sum(scaled, axis=1), 1.0, rtol=0.0, atol=1.0e-15)


def test_event_model_and_frozen_report_hashes_recompute() -> None:
    report = _report()
    event = ObservedTailEventModel.from_payload(report["event_model"])
    assert event.sha256 == report["event_model_sha256"]
    verify_report(report)
    assert report["gate_summary"]["passed"] == 28
    assert report["gate_summary"]["failed"] == 0
    assert report["calibration_workload"]["posterior_updates"] == 41 * 12 * 56 * 16
    assert report["pilot_workload"]["posterior_updates"] == 41 * 12 * 72 * 16
    assert report["threshold_selection"]["candidate_count"] == 1_728
    assert len(report["runtime_mutations"]) == 5
    assert len(report["semantic_mutations"]) == 14
    assert all(row["rejected"] for row in report["runtime_mutations"])
    assert all(row["rejected"] for row in report["semantic_mutations"])


def test_trace_seed_exactly_implements_preregistered_domain_separation() -> None:
    report = _report()
    contract = report["trace_seed_contract"]
    assert contract["protocol_id"] == protocol_payload()["protocol_id"]
    assert contract["streams"] == protocol_payload()["trace_schedule"]["streams"]
    cell = scenario_cells()[0]
    seeds = [_trace_seed(cell, 202607176001, name) for name in contract["streams"]]
    assert len(set(seeds)) == len(seeds)
    changed_cell = dict(cell)
    changed_cell["family"] = "variance_drift"
    assert _trace_seed(changed_cell, 202607176001, contract["streams"][0]) != seeds[0]


def test_pilot_baseline_winner_is_locked_and_not_oracle_or_legacy_cnn() -> None:
    report = _report()
    baseline = report["pilot_baseline_qualification"]
    summaries = baseline["method_summaries"]
    assert tuple(row["method_id"] for row in summaries) == BASELINE_METHODS
    ranked = min(
        summaries,
        key=lambda row: (
            row["equal_family_dynamic_average_ler"],
            row["dynamic_p95_window_ler"],
            row["dynamic_worst_window_ler"],
            row["update_macs"],
            row["wallclock_us_per_evaluated_decision"],
            row["method_id"],
        ),
    )
    assert baseline["selected_method_id"] == ranked["method_id"]
    assert baseline["selected_method_id"] == "ewma_adaptive_map"
    assert baseline["legacy_cnn_audit"]["eligible"] is False
    assert baseline["oracle_excluded"] is True
    assert baseline["formal_reselection_prohibited"] is True
    policy = report["pilot_policy_candidate_selection"]
    assert policy["adaptive_candidate_method_id"] == baseline["selected_method_id"]
    assert policy["policy_off_method_id"] == baseline["selected_method_id"]
    assert policy["fallback_method_id"] == "continuously_updated_validated_ewma_shadow_bank"
    assert policy["deployable_expert_banks"] == ["window_map", "ewma_adaptive_map"]
    router = policy["selected"]["dual_bank_router_cache_audit"]
    assert router["trajectory_count"] == 41 * 12
    assert router["candidate_count"] == policy["posterior_safe_candidate_count"]
    assert router["budget_audit"]["total_update_macs"] == 1_218
    assert router["budget_audit"]["passes"] is True
    assert router["truth_family_or_future_inputs"] is False
    assert policy["selected"]["pilot_ler_constraints_pass"] is True
    assert policy["formal_data_used"] is False
    assert (
        report["threshold_lock"]["lock_core"]["strongest_deployable_baseline"]
        == baseline["selected_method_id"]
    )


def test_source_data_independently_recomputes_pilot_baseline_ranking() -> None:
    report = _report()
    rows = []
    with Path(DEFAULT_SOURCE_DATA).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] == "pilot_baseline_cell":
                rows.append(json.loads(row["detail"]))
    assert len(rows) == len(BASELINE_METHODS) * 41 * 12
    expected = {
        row["method_id"]: row
        for row in report["pilot_baseline_qualification"]["method_summaries"]
    }
    recomputed = []
    for method in BASELINE_METHODS:
        method_rows = [row for row in rows if row["method_id"] == method]
        dynamic_families = sorted(
            {row["family"] for row in method_rows if row["family"] != "nominal_static"}
        )
        family_values = []
        for family in dynamic_families:
            family_rows = [row for row in method_rows if row["family"] == family]
            seed_values = []
            for seed in sorted({row["seed"] for row in family_rows}):
                seed_values.append(
                    np.mean([row["ler"] for row in family_rows if row["seed"] == seed])
                )
            family_values.append(np.mean(seed_values))
        window_counts = np.asarray(
            [
                count
                for row in method_rows
                if row["family"] != "nominal_static"
                for count in row["window_error_counts"]
            ]
        )
        values = (
            float(np.mean(family_values)),
            float(np.quantile(window_counts / 512.0, 0.95, method="higher")),
            float(np.max(window_counts) / 512.0),
        )
        stored = expected[method]
        assert values == pytest.approx(
            (
                stored["equal_family_dynamic_average_ler"],
                stored["dynamic_p95_window_ler"],
                stored["dynamic_worst_window_ler"],
            ),
            rel=0.0,
            abs=1.0e-15,
        )
        recomputed.append((*values, stored["update_macs"], stored["wallclock_us_per_evaluated_decision"], method))
    assert min(recomputed)[-1] == report["pilot_baseline_qualification"]["selected_method_id"]
