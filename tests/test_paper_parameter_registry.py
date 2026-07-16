from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "paper_parameter_registry.json"
MARKDOWN_PATH = ROOT / "docs" / "paper_parameter_registry.md"


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _by_name(registry: dict) -> dict[str, dict]:
    return {item["name"]: item for item in registry["parameters"]}


def _nested(mapping: dict, dotted_path: str):
    value = mapping
    for key in dotted_path.split("."):
        value = value[key]
    return value


def test_schema_has_all_required_categories_and_classifications(registry: dict) -> None:
    assert registry["task_id"] == "T1.4.4"
    assert registry["status"] == "frozen_contract"

    required_categories = {
        "protocol",
        "timing",
        "readout_category",
        "misclassification",
        "reset",
        "drift",
        "leakage",
        "squeezing_noise_ratio",
        "logical_metric",
    }
    required_classifications = {
        "literature_fact",
        "modeling_assumption",
        "secondary_reference",
        "pending_calibration",
    }
    assert set(registry["required_categories"]) == required_categories
    assert set(registry["classification_definitions"]) == required_classifications
    assert {item["category"] for item in registry["parameters"]} == required_categories
    assert {item["classification"] for item in registry["parameters"]} == required_classifications


def test_parameter_ids_and_contract_fields_are_complete(registry: dict) -> None:
    parameters = registry["parameters"]
    ids = [item["parameter_id"] for item in parameters]
    names = [item["name"] for item in parameters]
    assert len(parameters) == 51
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))

    required_fields = {
        "parameter_id",
        "category",
        "name",
        "value",
        "unit",
        "classification",
        "fact_scope",
        "source_id",
        "source_path",
        "line_start",
        "line_end",
        "expected_fragment",
        "normalization",
        "allowed_use",
        "forbidden_transfer",
        "calibration_gate",
    }
    for item in parameters:
        assert set(item) == required_fields, item["parameter_id"]
        assert item["unit"]
        assert item["fact_scope"]
        assert item["normalization"]
        assert item["allowed_use"]
        assert item["forbidden_transfer"]


def test_source_tiers_and_formal_doi_records_are_unambiguous(registry: dict) -> None:
    sources = {source["source_id"]: source for source in registry["sources"]}
    assert len(sources) == len(registry["sources"])

    formal_dois = []
    for source in sources.values():
        assert (ROOT / source["local_path"]).is_file()
        if source["publication_record"] == "formal":
            doi = source["doi"]
            assert doi == doi.lower()
            assert doi.startswith("10.")
            formal_dois.append(doi)
    assert len(formal_dois) == len(set(formal_dois))

    for item in registry["parameters"]:
        classification = item["classification"]
        if classification == "pending_calibration":
            continue
        tier = sources[item["source_id"]]["evidence_tier"]
        if classification == "literature_fact":
            assert tier == "primary_local", item["parameter_id"]
        elif classification == "secondary_reference":
            assert tier == "secondary_local", item["parameter_id"]
        elif classification == "modeling_assumption":
            assert tier == "project_local", item["parameter_id"]


def test_every_nonpending_anchor_matches_the_local_source(registry: dict) -> None:
    for item in registry["parameters"]:
        if item["classification"] == "pending_calibration":
            continue
        source_path = ROOT / item["source_path"]
        assert source_path.is_file(), item["parameter_id"]
        lines = source_path.read_text(encoding="utf-8").splitlines()
        start = item["line_start"]
        end = item["line_end"]
        assert isinstance(start, int) and isinstance(end, int)
        assert 1 <= start <= end <= len(lines)
        anchored_text = "\n".join(lines[start - 1 : end])
        assert item["expected_fragment"] in anchored_text, item["parameter_id"]


def test_pending_calibration_is_fail_closed_and_covers_every_category(registry: dict) -> None:
    pending = [
        item
        for item in registry["parameters"]
        if item["classification"] == "pending_calibration"
    ]
    assert len(pending) == 9
    assert {item["category"] for item in pending} == set(registry["required_categories"])
    for item in pending:
        assert item["value"] is None
        assert item["source_id"] is None
        assert item["source_path"] is None
        assert item["line_start"] is None
        assert item["line_end"] is None
        assert item["expected_fragment"] is None
        assert item["calibration_gate"]


def test_each_required_category_has_primary_literature_evidence(registry: dict) -> None:
    primary_categories = {
        item["category"]
        for item in registry["parameters"]
        if item["classification"] == "literature_fact"
    }
    assert primary_categories == set(registry["required_categories"])


def test_markdown_and_json_parameter_ids_are_exactly_synchronized(registry: dict) -> None:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    markdown_ids = re.findall(r"<!-- registry-id: ([A-Z0-9-]+) -->", markdown)
    json_ids = [item["parameter_id"] for item in registry["parameters"]]
    assert len(markdown_ids) == len(set(markdown_ids))
    assert set(markdown_ids) == set(json_ids)
    assert "## 6. 非 demo 审计结论" in markdown


def test_sivak_constituent_and_full_cycle_semantics_are_not_collapsed(registry: dict) -> None:
    items = _by_name(registry)
    cycle = items["sivak_composite_xz_cycle"]
    assert cycle["value"] == {
        "constituent_quadrature_steps": 2,
        "constituent_step_us": 4.924,
        "full_xz_cycle_us": 9.848,
    }
    assert cycle["value"]["full_xz_cycle_us"] == pytest.approx(
        2 * cycle["value"]["constituent_step_us"]
    )
    logical = items["sivak_peak_lifetime_gain_and_errors"]
    assert "per full X+Z QEC cycle" in logical["normalization"]

    puviani = items["puviani_model_cycle_duration_us"]
    project = items["project_fast_period_us"]
    assert puviani["value"]["half_cycle"] == project["value"] == 5.0
    assert puviani["classification"] == "literature_fact"
    assert project["classification"] == "modeling_assumption"
    assert "不得与 Sivak" in puviani["forbidden_transfer"]

    source_discrepancy = items["sivak_sbs_duration_source_discrepancy_ns"]
    assert source_discrepancy["value"] == {
        "prose_sbs": 1546,
        "table_layer_sum": 1548,
        "difference": 2,
    }
    assert source_discrepancy["calibration_gate"]


def test_readout_fidelity_does_not_invent_confusion_matrices(registry: dict) -> None:
    items = _by_name(registry)
    campagne = items["campagne_binary_readout_fidelity_bounds"]
    sivak = items["sivak_readout_fidelity_partial_matrix"]
    project = items["project_readout_confusion_matrix"]

    assert campagne["value"]["confusion_matrix"] is None
    assert campagne["value"]["integrated_readout_lower_bound"] == 0.99
    assert campagne["value"]["rabi_contrast_bound"] == 0.995
    assert sivak["value"]["F_g"] == 0.9997
    assert sivak["value"]["F_e"] == 0.9914
    assert sivak["value"]["F_f"] is None
    assert sivak["value"]["off_diagonal_transition_matrix"] is None
    assert project["classification"] == "pending_calibration"
    assert project["value"] is None


def test_puviani_idealizations_and_selection_caveat_are_explicit(registry: dict) -> None:
    items = _by_name(registry)
    assert items["puviani_readout_error_model"]["value"] is None
    assert items["puviani_numerical_reset"]["value"]["gate_or_pulse"] is False
    assert items["puviani_omitted_leakage_and_spam"]["value"] == {
        "leakage_modeled": False,
        "spam_modeled": False,
    }
    result = items["puviani_low_noise_simulated_lifetime_cycles"]
    assert result["value"]["selection"] == "best_of_20_agents"
    assert result["fact_scope"] == "numerical_simulation"
    assert "不得写成实验" in result["forbidden_transfer"]

    noise_levels = items["puviani_three_noise_levels_us"]
    assert noise_levels["value"]["low"] == {
        "cavity_Ts": 610,
        "ancilla_T1": 280,
        "ancilla_T2": 238,
    }
    assert noise_levels["value"]["medium"]["cavity_Ts"] == 490
    assert noise_levels["value"]["high"]["ancilla_T1"] == 50
    assert noise_levels["fact_scope"] == "theory_model_assumption"


def test_secondary_references_cannot_silently_become_primary(registry: dict) -> None:
    secondary = [
        item
        for item in registry["parameters"]
        if item["classification"] == "secondary_reference"
    ]
    assert {item["parameter_id"] for item in secondary} == {
        "SR-W01",
        "SR-W02",
        "SR-L01",
    }
    for item in secondary:
        assert item["calibration_gate"]
        assert "一手" in item["calibration_gate"] or "正式" in item["calibration_gate"]


def test_project_assumptions_match_the_live_yaml_config(registry: dict) -> None:
    config = yaml.safe_load(
        (ROOT / "cnn_fpga" / "config" / "hardware_hil.yaml").read_text(encoding="utf-8")
    )
    items = _by_name(registry)
    paths = {
        "project_fast_period_us": "runtime.t_fast_us",
        "project_histogram_window_size": "runtime.window_size",
        "project_slow_update_ms": "runtime.t_slow_update_ms",
        "project_sigma_measurement": "model.sigma_measurement",
        "project_sigma_ratio_p": "model.sigma_ratio_p",
        "project_ancilla_error_rate": "measurement.ancilla_error_rate",
        "project_measurement_efficiency": "measurement.measurement_efficiency",
        "project_effective_measurement_delta": "measurement.delta",
    }
    for name, path in paths.items():
        assert items[name]["classification"] == "modeling_assumption"
        assert items[name]["value"] == _nested(config, path)


def test_project_lattice_convention_matches_live_code(registry: dict) -> None:
    from physics.constants import LATTICE_CONST

    item = _by_name(registry)["project_lattice_const"]
    assert item["classification"] == "modeling_assumption"
    assert item["value"] == pytest.approx(math.sqrt(2.0 * math.pi))
    assert LATTICE_CONST == pytest.approx(item["value"])
    assert "scaling map" in item["forbidden_transfer"]


def test_global_rules_lock_version_timing_and_metric_transfer(registry: dict) -> None:
    rules = "\n".join(registry["global_transfer_rules"])
    assert "正式 DOI" in rules
    assert "off-diagonal" in rules
    assert "constituent/full-cycle" in rules
    assert "实测 artifact" in rules
