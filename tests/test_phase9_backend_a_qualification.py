from __future__ import annotations

import copy
import csv
from hashlib import sha256
import io
import json
from pathlib import Path

from cnn_fpga.benchmark import phase9_backend_a_qualification as artifact
from physics.phase9_backend_a import (
    BACKEND_A_ID,
    BackendAConfig,
    BackendAQualificationThresholds,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9/t9_2_2_backend_a.json"
REPORT = ROOT / "docs/t9_2_2_backend_a_qualification.json"
SOURCE_DATA = ROOT / "docs/t9_2_2_backend_a_qualification_source_data.csv"
MARKDOWN = ROOT / "docs/phase9_backend_a_qualification.md"
RELEASE_PIN = ROOT / "configs/phase9/t9_2_2_release_pin.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path: Path, relative: str):
    payload = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def test_canonical_report_rebuild_is_exact():
    assert artifact.build_report(root=ROOT, config_path=CONFIG) == load(REPORT)


def test_live_verify_passes_all_checks():
    report = load(REPORT)
    checks = artifact.verify_artifacts(
        root=ROOT,
        config_path=CONFIG,
        expected_analysis_sha256=report["analysis_sha256"],
    )
    assert checks
    assert all(checks.values())


def test_identity_and_parent_are_exact():
    report = load(REPORT)
    assert report["task_id"] == "T9.2.2"
    assert report["protocol_id"] == BACKEND_A_ID
    assert report["parent"]["task_id"] == "T9.2.1"
    parent_pin = load(ROOT / "configs/phase9/t9_2_1_release_pin.json")
    assert report["parent"]["analysis_sha256"] == parent_pin["analysis_sha256"]
    assert report["parent"]["all_parent_outcome_fields_null"] is True


def test_config_instantiates_exact_runtime_objects():
    config = load(CONFIG)
    runtime = BackendAConfig(**config["backend_parameters"])
    thresholds = BackendAQualificationThresholds(
        **config["qualification_thresholds"]
    )
    report = load(REPORT)
    assert runtime.semantic_sha256() == report["config"]["semantic_sha256"]
    assert thresholds.semantic_dict() == report["qualification_thresholds"]


def test_all_sources_are_live_hash_bound():
    report = load(REPORT)
    for value in report["implementation"].values():
        assert value == binding(ROOT / value["path"], value["path"])


def test_report_has_complete_gate_and_mutation_coverage():
    report = load(REPORT)
    assert report["gate_summary"] == {
        "passed": 20,
        "total": 20,
        "all_passed": True,
    }
    assert report["mutation_summary"] == {
        "detected": 20,
        "total": 20,
        "all_detected": True,
    }
    assert len(report["gate_definitions"]) == 20
    assert len(report["semantic_mutation_audit"]) == 20
    assert all(report["gates"].values())
    assert all(
        row["detected"] for row in report["semantic_mutation_audit"]
    )


def test_each_declared_mutation_really_fails_its_target_gate():
    report = artifact.build_report(root=ROOT, config_path=CONFIG)
    for _mutation_id, target_gate, mutate in artifact._mutations():
        changed = copy.deepcopy(report)
        mutate(changed)
        gates = artifact._evaluate_gates(
            changed,
            root=ROOT,
            config_path=CONFIG,
        )
        assert gates[target_gate] is False


def test_backend_qualification_checks_are_not_demo_sized():
    report = load(REPORT)
    checks = report["qualification"]["checks"]
    required = {
        "gksl_channel_cp",
        "gksl_channel_tp",
        "full_round_positive",
        "measurement_instrument_complete",
        "reset_instrument_complete",
        "zero_noise_idle_limit",
        "ideal_action_limit",
        "large_reset_limit",
        "reset_failure_preserves_f",
        "f_state_persistence",
        "iq_drives_measurement_backaction",
        "ramsey_syndrome_state_dependence",
        "syndrome_measurement_backacts_on_oscillator",
        "action_induces_physical_f_population",
        "action_changes_quantum_transition",
        "action_changes_latent_drift",
        "seed_determinism",
        "step_size_convergence",
        "fock_cutoff_convergence",
        "six_state_logical_projection",
    }
    assert len(checks) >= 27
    assert required <= set(checks)
    assert all(checks.values())


def test_convergence_metrics_satisfy_frozen_thresholds():
    report = load(REPORT)
    metrics = report["qualification"]["metrics"]
    thresholds = report["qualification_thresholds"]
    assert (
        metrics["step_size_16_vs_32_trace_distance"]
        <= thresholds["step_size_trace_distance"]
    )
    assert (
        metrics["step_size_error_ratio"]
        <= thresholds["step_size_error_ratio"]
    )
    assert (
        metrics["fock_cutoff_8_vs_12_trace_distance"]
        <= thresholds["fock_cutoff_trace_distance"]
    )
    assert (
        metrics["ideal_action_trace_distance"]
        <= thresholds["ideal_action_trace_distance"]
    )


def test_action_changes_physics_and_drift_under_common_randomness():
    report = load(REPORT)
    metrics = report["qualification"]["metrics"]
    thresholds = report["qualification_thresholds"]
    assert metrics["action_intervention_shared_exogenous"] is True
    assert (
        metrics["action_intervention_state_trace_distance"]
        > thresholds["action_state_trace_distance_minimum"]
    )
    assert (
        metrics["action_intervention_drift_l2"]
        > thresholds["action_drift_l2_minimum"]
    )
    assert (
        metrics["action_induced_f_population_difference"]
        > thresholds["action_induced_f_population_minimum"]
    )
    assert (
        metrics["syndrome_fock0_vs_fock1_level_tv"]
        > thresholds["syndrome_state_dependence_minimum"]
    )
    assert (
        metrics["syndrome_oscillator_backaction_trace_distance"]
        > thresholds["syndrome_backaction_trace_distance_minimum"]
    )


def test_model_contract_explicitly_rejects_label_noise_shortcut():
    report = load(REPORT)
    contract = report["model_contract"]
    assert "joint density matrix" in contract["state"]
    assert "no logical-label transition kernel" in contract["dynamics"]
    assert "Kraus" in contract["measurement"]
    assert "failed branch preserves e/f" in contract["reset"]
    assert "evaluator namespace only" in contract["logical_tracking"]


def test_truth_and_future_are_forbidden_transition_inputs():
    forbidden = set(load(REPORT)["namespace_contract"]["forbidden_transition_inputs"])
    assert {
        "logical_error",
        "logical_state_label",
        "future observation",
        "backend truth label",
        "controller-selected outcome",
    } == forbidden


def test_iq_boundary_remains_analog_pre_frontend():
    report = load(REPORT)
    assert report["gates"]["G18_analog_boundary"] is True
    assert "T9.2.6" in report["model_contract"]["iq_boundary"]
    assert "synthetic analog pre-frontend" in report["model_contract"]["iq_boundary"]


def test_all_downstream_claims_are_typed_null():
    report = load(REPORT)
    claims = report["qualification"]["claim_state"]
    assert len(claims) == 10
    assert all(value is None for value in claims.values())
    assert claims["backend_b_qualified"] is None
    assert claims["dual_backend_agreement"] is None
    assert claims["official_puviani_exact"] is None
    assert claims["external_sota"] is None
    assert claims["rank"] is None


def test_release_pin_matches_every_generated_artifact():
    pin = load(RELEASE_PIN)
    report = load(REPORT)
    assert pin["analysis_sha256"] == report["analysis_sha256"]
    assert pin["parent_analysis_sha256"] == report["parent"]["analysis_sha256"]
    for name, path in (
        ("report", REPORT),
        ("source_data", SOURCE_DATA),
        ("markdown", MARKDOWN),
    ):
        relative = path.relative_to(ROOT).as_posix()
        assert pin[name] == binding(path, relative)
    assert pin["dependency_and_test_bindings"] == {
        key: value
        for key, value in report["implementation"].items()
        if key not in {"backend", "generator"}
    }


def test_source_data_has_all_metrics_checks_gates_mutations_and_nulls():
    rows = list(
        csv.DictReader(io.StringIO(SOURCE_DATA.read_text(encoding="utf-8")))
    )
    report = load(REPORT)
    sections = {}
    for row in rows:
        sections[row["section"]] = sections.get(row["section"], 0) + 1
    assert sections["metric"] == len(report["qualification"]["metrics"])
    assert sections["check"] == len(report["qualification"]["checks"])
    assert sections["gate"] == 20
    assert sections["mutation"] == 20
    assert sections["typed_null_claim"] == 10


def test_markdown_discloses_scope_and_negative_boundaries():
    text = MARKDOWN.read_text(encoding="utf-8")
    assert "不是给标签加独立噪声" in text
    assert "synthetic analog pre-frontend" in text
    assert "backend B、双后端对拍" in text
    assert "official Puviani" in text
    assert "全部保持 `null`" in text


def test_analysis_hash_is_reproducible_and_json_has_no_nan():
    report = load(REPORT)
    rebuilt = artifact.build_report(root=ROOT, config_path=CONFIG)
    assert rebuilt["analysis_sha256"] == report["analysis_sha256"]
    encoded = json.dumps(report, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
