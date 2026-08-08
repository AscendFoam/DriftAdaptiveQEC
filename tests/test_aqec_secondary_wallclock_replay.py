from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from cnn_fpga.benchmark import aqec_secondary_wallclock_replay as audit


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_18_1_aqec_common_wallclock_replay.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_recomputes_all_gates_and_mutations_in_default_python() -> None:
    report = _report()
    audit.verify_report(report)
    assert report["verdict"] == audit.VERDICT
    assert report["gate_summary"] == {"passed": 16, "failed": []}
    assert report["semantic_mutation_audit"]["count"] == 16
    assert report["semantic_mutation_audit"]["detected"] == 16
    assert all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"])


def test_registered_seed_universe_and_cells_are_complete() -> None:
    report = _report()
    expected_seeds = list(range(61_810_001, 61_810_025))
    assert len(report["cells"]) == 6
    assert {(cell["cutoff"], cell["noise_profile"]) for cell in report["cells"]} == {
        (cutoff, noise) for cutoff in (12, 16) for noise in ("high", "medium", "low")
    }
    assert all([row["seed"] for row in cell["seed_records"]] == expected_seeds for cell in report["cells"])
    assert report["project_result"]["seed_clusters"] == 144


def test_noise_realizations_are_deterministic_physical_and_common_random() -> None:
    report = _report()
    fingerprints = []
    for cell in report["cells"]:
        for row in cell["seed_records"]:
            assert row["noise"] == audit.sampled_lifetimes(row["seed"], cell["cutoff"], cell["noise_profile"])
            assert row["noise"]["ancilla_t2_us"] <= 2.0 * row["noise"]["ancilla_t1_us"]
            assert {row["anchors"][anchor]["noise_fingerprint"] for anchor in audit.ANCHORS} == {row["noise"]["fingerprint"]}
            fingerprints.append(row["noise"]["fingerprint"])
    assert len(set(fingerprints)) == len(fingerprints) == 144


def test_all_raw_curves_reach_common_horizon_and_registered_grids() -> None:
    for cell in _report()["cells"]:
        for row in cell["seed_records"]:
            idle = row["anchors"]["idle_memory"]
            feedback = row["anchors"]["measurement_feedback"]
            autonomous = row["anchors"]["autonomous"]
            assert len(idle["time_us"]) == len(feedback["time_us"]) == 71
            assert len(autonomous["time_us"]) == 101
            assert idle["time_us"][-1] == feedback["time_us"][-1] == autonomous["time_us"][-1] == 700.0
            assert np.allclose(np.diff(feedback["time_us"]), 10.0)
            assert np.allclose(np.diff(autonomous["time_us"]), 7.0)
            for payload in (idle, feedback, autonomous):
                assert set(payload["curves"]) == set(audit.CURVE_METRICS)
                assert all(len(values) == len(payload["time_us"]) for values in payload["curves"].values())


def test_lifetime_survival_and_gain_recompute_from_raw_curves() -> None:
    for cell in _report()["cells"]:
        for row in cell["seed_records"]:
            idle_lifetime = row["anchors"]["idle_memory"]["metrics"]["logical_lifetime_us"]
            for anchor in audit.ANCHORS:
                payload = row["anchors"][anchor]
                area = audit.area_equivalent_lifetime(payload["time_us"], payload["curves"]["logical_z_signal"])
                assert payload["area_lifetime"] == area
                assert payload["metrics"]["logical_lifetime_us"] == area["area_equivalent_lifetime_us"]
                assert payload["metrics"]["logical_lifetime_cycles"] == area["area_equivalent_lifetime_us"] / payload["cycle_duration_us"]
                assert payload["metrics"]["final_code_survival"] == payload["curves"]["code_survival"][-1]
                assert payload["metrics"]["lifetime_gain_ratio"] == payload["metrics"]["logical_lifetime_us"] / idle_lifetime


def test_fit_diagnostics_cover_logical_survival_and_fidelity_without_hiding_bad_fit() -> None:
    low_r2_count = 0
    for cell in _report()["cells"]:
        for row in cell["seed_records"]:
            for payload in row["anchors"].values():
                assert set(payload["fit_diagnostics"]) == {"logical_z_signal", "code_survival", "fidelity"}
                for metric, fit in payload["fit_diagnostics"].items():
                    assert fit == audit.exponential_fit_diagnostic(payload["time_us"], payload["curves"][metric])
                    assert fit["points"] >= 71
                    assert fit["status"] in {"DECAY_FIT", "NONDECAYING_FIT"}
                    low_r2_count += fit["r_squared"] is not None and fit["r_squared"] < 0.5
    assert low_r2_count > 0  # non-exponential curves remain visible rather than fit-selected away


def test_paired_bootstrap_and_ordering_reversal_recompute() -> None:
    report = _report()
    for index, cell in enumerate(report["cells"]):
        assert cell["summary"] == audit._cell_summary(cell["seed_records"], index)
        assert cell["summary"]["ordering_reversal"] == {
            "definition": "autonomous lifetime is higher in protocol cycles but lower in common-wall-clock microseconds",
            "count": 24,
            "total": 24,
            "fraction": 1.0,
        }
        assert all(row["bootstrap_reps"] == 20_000 for row in cell["summary"]["paired"].values())


def test_negative_project_result_is_preserved_without_universal_gain_claim() -> None:
    report = _report()
    assert report["project_result"]["cells_with_gain_ci_lower_above_one"] == {
        "measurement_feedback": 0,
        "autonomous": 0,
    }
    assert report["project_result"]["ordering_reversal_cells"] == 6
    assert report["project_result"]["universal_20_percent_claim"] is False
    for cell in report["cells"]:
        paired = cell["summary"]["paired"]
        assert paired["measurement_vs_idle_lifetime_gain"]["ci95"][1] < 1.0
        assert paired["autonomous_vs_idle_lifetime_gain"]["ci95"][1] < 1.0
        assert paired["autonomous_vs_measurement_lifetime_us_ratio"]["ci95"][1] < 1.0


def test_event_accounting_and_unavailable_fields_are_protocol_native() -> None:
    for cell in _report()["cells"]:
        for row in cell["seed_records"]:
            idle, feedback, autonomous = (row["anchors"][name] for name in audit.ANCHORS)
            assert idle["metrics"]["measurements_per_100us"] == 0.0
            assert idle["metrics"]["resets_per_100us"] == 0.0
            assert feedback["metrics"]["measurements_per_100us"] == feedback["metrics"]["resets_per_100us"] == 20.0
            assert autonomous["metrics"]["measurements_per_100us"] == 0.0
            assert autonomous["metrics"]["resets_per_100us"] == 200.0 / 7.0
            assert all(value is None for payload in (idle, feedback, autonomous) for value in payload["unavailable_fields"].values())


def test_lachance_metrics_remain_literature_only_and_official_reproduction_blocked() -> None:
    report = _report()
    metrics = {row["metric_id"]: row for row in report["literature_only_metrics"]}
    assert metrics["method_a_gain"]["value"] == 1.14
    assert metrics["method_a_gain"]["uncertainty"] == 0.18
    assert metrics["method_b_gain"]["value"] == 1.14
    assert metrics["method_b_gain"]["uncertainty"] == 0.16
    assert all(row["evidence_grade"] == "LITERATURE_ONLY" for row in metrics.values())
    assert report["official_protocol_reproduction"] == {
        "state": "BLOCKED_OFFICIAL_PROTOCOL_REPRODUCTION",
        "official_code_available": False,
        "paper_native_reservoir_adapter": False,
        "project_replay_may_substitute": False,
    }


def test_source_data_exact_counts_and_evidence_states() -> None:
    report = _report()
    with (ROOT / report["source_data"]["path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 144_152
    counts = {kind: sum(row["record_type"] == kind for row in rows) for kind in {"noise", "curve", "seed_metric", "paired", "literature", "boundary"}}
    assert counts == {"noise": 576, "curve": 139_968, "seed_metric": 3_024, "paired": 576, "literature": 6, "boundary": 2}
    assert {row["value_state"] for row in rows if row["record_type"] == "literature"} == {"LITERATURE_ONLY"}
    assert {row["value_state"] for row in rows if row["record_type"] == "boundary"} == {"N_A_NOT_APPLICABLE", "BLOCKED"}


def test_runtime_memory_and_raw_reprocess_provenance_are_explicit() -> None:
    report = _report()
    budget = report["execution_budget_audit"]
    assert budget["runtime_seconds"] < budget["runtime_budget_seconds"] == 14_400
    assert budget["accounted_peak_memory_bytes"] == max(budget["peak_device_memory_bytes"], budget["peak_host_working_set_bytes"])
    assert budget["peak_host_working_set_bytes"] > 0
    assert "peak_wset" in budget["host_peak_observation"]
    assert budget["accounted_peak_memory_bytes"] < budget["memory_budget_bytes"] == 12 * (1 << 30)
    assert report["analysis_reuse"]["state"] in {"RAW_CURVE_REPROCESS_NO_SIMULATION_CHANGE", "FRESH_SIMULATION"}
    if report["analysis_reuse"]["state"] == "RAW_CURVE_REPROCESS_NO_SIMULATION_CHANGE":
        assert len(report["analysis_reuse"]["input_report_sha256"]) == 64
        assert len(report["analysis_reuse"]["reused_raw_cells_sha256"]) == 64


def test_live_gates_reject_forged_lifetime_and_official_upgrade() -> None:
    report = _report()
    forged_lifetime = deepcopy(report)
    forged_lifetime["cells"][0]["seed_records"][0]["anchors"]["measurement_feedback"]["metrics"]["logical_lifetime_us"] = 1e9
    assert not audit.evaluate_gates(forged_lifetime)["G10_lifetime_survival_and_fit_metrics_recompute_from_every_raw_curve"]
    forged_official = deepcopy(report)
    forged_official["official_protocol_reproduction"]["project_replay_may_substitute"] = True
    assert not audit.evaluate_gates(forged_official)["G13_official_reservoir_reproduction_and_energy_latency_fields_fail_closed"]


def test_all_artifact_bindings_and_markdown_boundaries_are_live() -> None:
    report = _report()
    for binding in report["bindings"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert audit._sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["bytes"]
    text = (ROOT / "docs" / "aqec_common_wallclock_replay.md").read_text(encoding="utf-8")
    assert "ordering-reversal cells：6/6" in text
    assert "不是 Lachance 2024" in text
    assert "N/A 而非 0" in text
    assert "pulse energy 与 control-duty" in text
