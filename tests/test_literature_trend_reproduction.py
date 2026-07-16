from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.literature_trend_reproduction import (
    ARTIFACT_BINDINGS,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    STATUS_VALUES,
    build_registry,
    implementation_sha256,
    load_bound_artifacts,
    write_artifacts,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def _targets(payload: dict) -> dict[str, dict]:
    return {row["target_id"]: row for row in payload["targets"]}


def test_committed_artifact_is_current_complete_and_source_bound() -> None:
    payload = _artifact()
    assert payload["task_id"] == "T5.0.1"
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["gate_summary"] == {
        "passed": len(payload["gates"]),
        "total": len(payload["gates"]),
        "failed": [],
    }
    assert len(payload["gates"]) == 17 and all(payload["gates"].values())
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()


def test_seven_primary_sources_and_local_anchors_are_explicit_and_current() -> None:
    payload = _artifact()
    sources = {row["source_id"]: row for row in payload["sources"]}
    assert len(sources) == 7
    assert sources["SRC-CAMPAGNE-2020"]["identifier"] == "doi:10.1038/s41586-020-2603-3"
    assert sources["SRC-SIVAK-2023"]["identifier"] == "doi:10.1038/s41586-023-05782-6"
    assert sources["SRC-PUVIANI-PRL-2025"]["identifier"] == "doi:10.1103/PhysRevLett.134.020601"
    assert sources["SRC-FONTBOTE-2026"]["identifier"] == "arXiv:2605.08009v1"
    assert all(row["official_url"].startswith("https://") for row in sources.values())
    assert len(payload["local_source_anchors"]) == 8
    for anchor in payload["local_source_anchors"]:
        assert anchor["passed"] is True
        path = Path(anchor["path"])
        assert path.is_file()
        actual = path.read_text(encoding="utf-8").splitlines()[anchor["line"] - 1]
        assert anchor["fragment"] in actual
        assert anchor["actual_line_sha256"] == hashlib.sha256(actual.encode("utf-8")).hexdigest()


def test_all_bound_project_artifacts_are_hash_current_and_machine_pass() -> None:
    payload = _artifact()
    assert set(payload["artifact_bindings"]) == set(ARTIFACT_BINDINGS)
    for task_id, path in ARTIFACT_BINDINGS.items():
        binding = payload["artifact_bindings"][task_id]
        assert binding["path"] == str(path)
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert binding["machine_pass"] is True


def test_fourteen_targets_have_controlled_statuses_and_split_contracts() -> None:
    payload = _artifact()
    targets = payload["targets"]
    assert len(targets) == 14
    assert len({row["target_id"] for row in targets}) == 14
    assert {row["current_status"] for row in targets} <= STATUS_VALUES
    required_uses = {
        "calibration_only",
        "independent_holdout",
        "future_holdout_preregistered",
        "reference_only_no_model_selection",
        "reporting_template_only",
    }
    assert {row["calibration_or_holdout_use"] for row in targets} == required_uses
    for row in targets:
        assert row["reference_target"]
        assert row["tolerance_rule"]
        assert row["next_gate"]
        assert row["prohibited_transfer"]
        if row["calibration_or_holdout_use"] != "calibration_only":
            assert row["model_selection_access"] is False


def test_pending_and_reference_rows_are_not_counted_as_reproduction_passes() -> None:
    payload = _artifact()
    by_status = payload["coverage_summary"]["by_status"]
    assert by_status["REGISTERED_PENDING"] == 5
    assert by_status["REFERENCE_ONLY"] == 2
    assert by_status["REPORTING_TEMPLATE_ONLY"] == 1
    assert payload["coverage_summary"]["current_pass_like_count"] == 6
    assert "does not mean every registered literature trend" in payload["completion_semantics"]
    pending = [row for row in payload["targets"] if row["current_status"] == "REGISTERED_PENDING"]
    assert all(not row["evidence_artifacts"] or row["current_observation"].get("secondary_protocol_executable") is False for row in pending)


def test_campagne_and_sivak_targets_separate_reference_from_project_evidence() -> None:
    targets = _targets(_artifact())
    structure = targets["LT-2020-STRUCTURE"]
    assert structure["current_status"] == "STRUCTURE_IMPLEMENTED_NOT_NUMERIC_REPRODUCTION"
    assert structure["reference_target"]["peak_sharpen_rounds"] == 2
    assert structure["reference_target"]["envelope_trim_rounds"] == 2
    lifetime = targets["LT-2020-QEC-ON-OFF"]
    assert lifetime["reference_target"]["qec_on_us"] == {"X": 275, "Y": 160, "Z": 275}
    assert lifetime["tolerance_rule"]["match_external_microseconds"] is False
    timing = targets["LT-2023-GAIN-TIMING"]
    assert timing["reference_target"]["full_cycle_us"] == pytest.approx(9.848)
    assert timing["current_observation"]["project_device_timing_measured"] is False
    displacement = targets["LT-2023-DISPLACEMENT"]
    assert displacement["current_observation"]["peak_location"] == pytest.approx(0.25)
    assert displacement["current_status"] == "QUALIFIED_DIRECTIONAL_PASS"


def test_nmf_direction_is_model_specific_and_cutoff_reversal_is_preserved() -> None:
    row = _targets(_artifact())["LT-2025-NMF-DIRECTION"]
    assert row["current_observation"]["ci95_low"] == pytest.approx(0.08416109825708099)
    assert row["current_observation"]["ci95_high"] == pytest.approx(0.32806715194289604)
    assert row["current_observation"]["later_exact_budget_MF_ordering_reverses_across_cutoffs"] is True
    prohibited = " ".join(row["prohibited_transfer"])
    assert "universal NMF-over-MF" in prohibited
    assert "paper-amplitude" in prohibited
    horizon = _targets(_artifact())["LT-2025-NMF-HORIZON"]
    assert horizon["current_status"] == "REGISTERED_PENDING"
    assert horizon["reference_target"]["required_project_sweeps"] == [1000, 100000, 1000000]


def test_secondary_knill_qunaught_and_psteane_never_enter_main_ranking() -> None:
    targets = _targets(_artifact())
    ids = {
        "LT-2025-KNILL-EQUIVALENCE",
        "LT-2025-QUNAUGHT-SQUEEZING",
        "LT-2026-PSTEANE-CONDITION",
        "LT-2026-PSTEANE-NOISE-RATIO",
    }
    for target_id in ids:
        row = targets[target_id]
        assert row["hierarchy_role"] == "secondary_reproduction"
        assert row["current_status"] == "REGISTERED_PENDING"
        assert row["current_observation"]["secondary_protocol_executable"] is False
        assert "sBs main ranking" in " ".join(row["prohibited_transfer"])
    assert targets["LT-2025-KNILL-EQUIVALENCE"]["tolerance_rule"]["max_absolute_error"] == 1e-8
    assert targets["LT-2026-PSTEANE-CONDITION"]["reference_target"]["condition"] == "2a=b"


def test_noise_transfer_positive_and_negative_domains_are_both_frozen() -> None:
    targets = _targets(_artifact())
    high = targets["LT-2024-NOISE-TRANSFER-HIGH"]
    assert high["current_status"] == "QUALIFIED_DIRECTIONAL_PASS"
    assert high["current_observation"]["noise_vs_direct_q_ler_gap"] == pytest.approx(3.926171650157906e-05)
    assert high["current_observation"]["effective_noise_z_score"] <= 2.0
    assert high["current_observation"]["canonical_qp_gap"] <= 1e-6
    low = targets["LT-2024-NOISE-TRANSFER-LOW"]
    assert low["current_status"] == "NEGATIVE_BOUNDARY_VERIFIED"
    assert low["current_observation"]["noise_vs_direct_q_ler_gap"] >= 0.01
    assert low["current_observation"]["minimum_clipping_ratio"] < 0.5


def test_trapped_ion_result_is_only_a_complete_reporting_template() -> None:
    row = _targets(_artifact())["LT-2026-TRAPPED-ION-REPORT"]
    fields = set(row["reference_target"]["required_fields"])
    assert fields == {
        "Pauli_resolved_on_off_lifetimes",
        "ratio_with_uncertainty",
        "wall_clock_per_round",
        "reset_recoil",
        "parallel_control_cost",
    }
    external = row["reference_target"]["external_context"]
    assert external["Bell_fidelity"] == "0.69(1)"
    assert external["Pauli_lifetime_ms"] == {
        "XX": {"on": "5.0(7)", "off": "2.4(2)", "ratio": "2.1(3)"},
        "YY": {"on": "3.8(9)", "off": "2.3(4)", "ratio": "1.7(5)"},
        "ZZ": {"on": "5.3(8)", "off": "2.3(2)", "ratio": "2.3(4)"},
    }
    assert external["mean_lifetime_gain"] == "2.0(2)"
    assert external["round_us"] == 500
    assert row["current_status"] == "REPORTING_TEMPLATE_ONLY"
    assert row["model_selection_access"] is False
    assert "cannot be transferred" in " ".join(row["prohibited_transfer"])


def test_source_data_is_a_complete_52_row_ledger() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"] == 52
    counts = {row_type: sum(row["row_type"] == row_type for row in rows) for row_type in {row["row_type"] for row in rows}}
    assert counts == {"source": 7, "source_anchor": 8, "artifact_binding": 6, "target": 14, "gate": 17}
    target_rows = [row for row in rows if row["row_type"] == "target"]
    assert sum(row["passed"] == "True" for row in target_rows) == 6


def test_human_table_preserves_status_and_nontransfer_boundaries() -> None:
    text = Path("docs/literature_trend_reproduction_table.md").read_text(encoding="utf-8")
    for token in (
        "14 个趋势目标",
        "REGISTERED_PENDING",
        "3.92617e-5",
        "0.015408",
        "exact-budget MF",
        "5.0(7)/3.8(9)/5.3(8) ms",
        "不是 FPGA 实测",
    ):
        assert token in text


def test_registry_contract_is_deterministic_and_missing_inputs_fail_closed() -> None:
    first = build_registry()
    second = build_registry()
    assert first["registry_contract_sha256"] == second["registry_contract_sha256"]
    artifacts = load_bound_artifacts()
    artifacts.pop("T2.3.3")
    with pytest.raises(ValueError, match="missing bound artifacts"):
        build_registry(artifacts)


def test_mutated_high_squeezing_boundary_fails_without_promoting_pending_rows() -> None:
    artifacts = copy.deepcopy(load_bound_artifacts())
    artifacts["T2.3.3"]["maximum_high_squeezing_noise_syndrome_q_ler_gap"] = 1e-2
    result = build_registry(artifacts)
    assert result["status"] == "FAIL"
    assert result["gates"]["high_squeezing_noise_transfer_gate_matches_current_artifact"] is False
    assert all(row["current_status"] == "REGISTERED_PENDING" for row in result["targets"] if row["current_observation"].get("secondary_protocol_executable") is False)


def test_writer_round_trip_preserves_machine_status_and_csv_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "registry.json"
    source_data = tmp_path / "source.csv"
    payload = write_artifacts(artifact, source_data)
    reloaded = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == reloaded["status"] == "PASS"
    assert reloaded["source_data"]["row_count"] == 52
    assert reloaded["source_data"]["sha256"] == hashlib.sha256(source_data.read_bytes()).hexdigest()
