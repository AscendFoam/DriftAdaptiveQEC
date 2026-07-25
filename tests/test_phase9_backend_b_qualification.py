from __future__ import annotations

import copy
import csv
from hashlib import sha256
import io
import json
from pathlib import Path

from cnn_fpga.benchmark import phase9_backend_b_qualification as artifact
from physics.phase9_backend_b import (
    BACKEND_B_ID,
    BackendBConfig,
    BackendBQualificationThresholds,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9/t9_2_3_backend_b.json"
REPORT = ROOT / "docs/t9_2_3_backend_b_qualification.json"
SOURCE_DATA = ROOT / "docs/t9_2_3_backend_b_qualification_source_data.csv"
MARKDOWN = ROOT / "docs/phase9_backend_b_qualification.md"
RELEASE_PIN = ROOT / "configs/phase9/t9_2_3_release_pin.json"


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


def test_live_verify_passes_every_check():
    report = load(REPORT)
    checks = artifact.verify_artifacts(
        root=ROOT,
        config_path=CONFIG,
        expected_analysis_sha256=report["analysis_sha256"],
    )
    assert checks and all(checks.values())


def test_identity_parent_and_backend_a_bindings_are_exact():
    report = load(REPORT)
    assert report["task_id"] == "T9.2.3"
    assert report["protocol_id"] == BACKEND_B_ID
    assert report["parent"]["task_id"] == "T9.2.1"
    assert report["comparison_backend_a"]["task_id"] == "T9.2.2"
    assert report["parent"]["all_outcome_fields_null"] is True
    assert report["comparison_backend_a"]["all_outcome_fields_null"] is True


def test_config_instantiates_exact_runtime_objects():
    config = load(CONFIG)
    runtime = BackendBConfig(**config["backend_parameters"])
    thresholds = BackendBQualificationThresholds(
        **config["qualification_thresholds"]
    )
    report = load(REPORT)
    assert runtime.semantic_sha256() == report["config"]["semantic_sha256"]
    assert thresholds.semantic_dict() == report["qualification_thresholds"]


def test_all_sources_are_live_hash_bound():
    report = load(REPORT)
    for value in report["implementation"].values():
        assert value == binding(ROOT / value["path"], value["path"])


def test_static_isolation_manifest_excludes_forbidden_reuse():
    isolation = load(REPORT)["independence_manifest"]
    assert isolation["static_ast_isolation_passed"] is True
    assert isolation["forbidden_runtime_import_hits"] == []
    assert isolation["forbidden_source_token_hits"] == []
    assert isolation["backend_a_runtime_imported"] is False
    assert isolation["transition_kernel_reused"] is False
    assert isolation["iq_sampler_reused"] is False
    assert isolation["logical_projector_reused"] is False
    assert isolation["rng_stream_reused"] is False
    assert isolation["precomputed_truth_reused"] is False


def test_report_has_complete_gate_and_mutation_coverage():
    report = load(REPORT)
    assert report["gate_summary"] == {
        "passed": 22,
        "total": 22,
        "all_passed": True,
    }
    assert report["mutation_summary"] == {
        "detected": 22,
        "total": 22,
        "all_detected": True,
    }
    assert len(report["gate_definitions"]) == 22
    assert len(report["semantic_mutation_audit"]) == 22


def test_each_declared_mutation_really_fails_target_gate():
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


def test_qualification_checks_cover_real_physics_not_label_noise():
    checks = load(REPORT)["qualification"]["checks"]
    required = {
        "split_channel_cp",
        "split_channel_tp",
        "full_round_positive",
        "analytic_channels_complete",
        "pure_loss_closed_form",
        "relaxation_closed_form",
        "iq_kraus_backaction",
        "ramsey_syndrome_state_dependence",
        "syndrome_backacts_on_oscillator",
        "action_induces_f_population",
        "action_changes_quantum_state",
        "action_changes_drift",
        "seed_determinism",
        "seed_sensitivity",
        "split_step_convergence",
        "fock_cutoff_convergence",
        "independent_six_state_logical_projection",
    }
    assert len(checks) >= 28
    assert required <= set(checks)
    assert all(checks.values())


def test_closed_form_and_convergence_metrics_satisfy_frozen_thresholds():
    report = load(REPORT)
    metrics = report["qualification"]["metrics"]
    thresholds = report["qualification_thresholds"]
    assert (
        metrics["analytic_loss_mean_error"]
        <= thresholds["analytic_loss_mean_error"]
    )
    assert (
        metrics["analytic_relaxation_population_error"]
        <= thresholds["analytic_relaxation_error"]
    )
    assert metrics["split_error_ratio"] <= thresholds["split_ratio"]
    assert (
        metrics["split_16_vs_32_trace_distance"]
        <= thresholds["split_distance"]
    )
    assert (
        metrics["fock_cutoff_8_vs_12_trace_distance"]
        <= thresholds["cutoff_distance"]
    )


def test_solver_rng_likelihood_and_logical_paths_are_explicit():
    report = load(REPORT)
    toolchain = report["toolchain"]
    assert "DENSE_EXPM_STRANG" in toolchain["solver_id"]
    assert "PYTHON_RANDOM_BOX_MULLER" in toolchain["rng_id"]
    assert "GAUSSIAN_LOG_LIKELIHOOD" in toolchain["likelihood_id"]
    assert "SQUEEZED_COHERENT_COMB" in toolchain["logical_id"]
    assert "expm_multiply" not in toolchain["solver"]


def test_truth_future_and_backend_a_outputs_are_forbidden_inputs():
    forbidden = set(
        load(REPORT)["namespace_contract"]["forbidden_transition_inputs"]
    )
    assert {
        "logical_error",
        "logical_state_label",
        "future observation",
        "backend truth label",
        "backend A output",
        "controller-selected outcome",
    } == forbidden


def test_all_downstream_claims_remain_typed_null():
    claims = load(REPORT)["qualification"]["claim_state"]
    assert len(claims) == 10
    assert all(value is None for value in claims.values())
    assert claims["backend_a_b_agreement"] is None
    assert claims["official_puviani_exact"] is None
    assert claims["puviani_nmf_surpass"] is None
    assert claims["external_sota"] is None
    assert claims["rank"] is None


def test_release_pin_matches_all_artifacts_and_dependencies():
    pin = load(RELEASE_PIN)
    report = load(REPORT)
    assert pin["analysis_sha256"] == report["analysis_sha256"]
    assert pin["parent_analysis_sha256"] == report["parent"]["analysis_sha256"]
    assert (
        pin["backend_a_analysis_sha256"]
        == report["comparison_backend_a"]["analysis_sha256"]
    )
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


def test_source_data_contains_every_evidence_class():
    rows = list(
        csv.DictReader(io.StringIO(SOURCE_DATA.read_text(encoding="utf-8")))
    )
    report = load(REPORT)
    sections: dict[str, int] = {}
    for row in rows:
        sections[row["section"]] = sections.get(row["section"], 0) + 1
    assert sections["metric"] == len(report["qualification"]["metrics"])
    assert sections["check"] == len(report["qualification"]["checks"])
    assert sections["gate"] == 22
    assert sections["mutation"] == 22
    assert sections["typed_null_claim"] == 10


def test_markdown_discloses_independence_and_negative_boundaries():
    text = MARKDOWN.read_text(encoding="utf-8")
    assert "独立实现证据" in text
    assert "Box–Muller" in text
    assert "forbidden import" in text
    assert "不证明 A/B" in text
    assert "official Puviani exact/surpass" in text
    assert "全部保持 `null`" in text


def test_analysis_hash_is_reproducible_and_json_has_no_nonfinite_values():
    report = load(REPORT)
    rebuilt = artifact.build_report(root=ROOT, config_path=CONFIG)
    assert rebuilt["analysis_sha256"] == report["analysis_sha256"]
    encoded = json.dumps(report, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
