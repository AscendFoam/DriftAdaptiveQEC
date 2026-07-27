from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_high_cutoff_design_diagnostic as subject
from cnn_fpga.benchmark.phase9_paired_cluster_uq import NormUCB


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _activate_verified_diagnostic_dependencies() -> None:
    subject.__verified_source_sha256__ = subject._DIAGNOSTIC_SOURCE_SHA256_AT_IMPORT
    subject.__verified_bootstrap_contract__ = subject.VERIFIED_LOADER_CONTRACT
    subject.__verified_external_launcher_sha256__ = subject.EXTERNAL_LAUNCHER_SHA256
    subject.__verified_bootstrap_source_binding__ = {
        "path": "bootstrap.py",
        "bytes": 1,
        "sha256": "b" * 64,
    }
    subject.__verified_launch_meta_binding__ = {
        "path": "launch.json",
        "bytes": 1,
        "sha256": "c" * 64,
    }
    launch_meta = {
        "task_id": subject.TASK_ID,
        "schema_version": subject.LAUNCH_META_SCHEMA,
        "mode": "diagnostic",
        "external_launcher_sha256": subject.EXTERNAL_LAUNCHER_SHA256,
        "launcher_assurance": dict(subject.LAUNCHER_ASSURANCE),
        "isolation_flags": ["-I", "-S"],
        "bootstrap": dict(subject.__verified_bootstrap_source_binding__),
        "bootstrap_load_protocol": "read_once_sha256_then_compile_exec",
        "child_process_policy": "same_verified_process_thread_workers_only",
        "qualified_claim": None,
        "downstream_release": False,
    }
    launch_meta["analysis_sha256"] = subject._sha(launch_meta)
    subject.__verified_launch_meta_payload__ = launch_meta
    subject._activate_verified_diagnostic_modules(ROOT)


def _fake_ucb(left, right, **kwargs) -> NormUCB:
    count = len(left)
    return NormUCB(
        estimate=0.01,
        raw_radius=0.02,
        calibrated_radius=0.02,
        quantization_bound=float(
            np.mean(kwargs.get("quantization_bounds", np.zeros(count)))
        ),
        upper_bound=0.03,
        confidence=float(kwargs["confidence"]),
        multiplier_replicates=int(kwargs["multiplier_replicates"]),
        cluster_count=count,
        calibration_factor=float(kwargs["calibration_factor"]),
        seed=int(kwargs["seed"]),
    )


def test_launch_assurance_cannot_be_rehashed_into_stronger_claim(monkeypatch) -> None:
    upgraded = json.loads(json.dumps(subject.__verified_launch_meta_payload__))
    upgraded["launcher_assurance"][
        "cryptographic_process_origin_attestation"
    ] = "VERIFIED"
    upgraded.pop("analysis_sha256")
    upgraded["analysis_sha256"] = subject._sha(upgraded)
    monkeypatch.setattr(subject, "__verified_launch_meta_payload__", upgraded)

    with pytest.raises(RuntimeError, match="trusted-operator bootstrap"):
        subject._require_verified_self_import()


def _fixture():
    states = ("0", "1", "+", "-", "+i", "-i")
    rows = []
    densities = {}
    for cutoff in (2, 3):
        for backend in ("A", "B"):
            for position in range(72):
                state = states[position % 6]
                for round_index in range(12):
                    row_id = f"c{cutoff}-{backend}-p{position}-r{round_index}"
                    terminal = round_index == 11
                    rows.append(
                        {
                            "row_id": row_id,
                            "cutoff": cutoff,
                            "scenario": "step",
                            "backend": backend,
                            "logical_label": state,
                            "seed_position": position,
                            "round_index": round_index,
                            "terminal_round": terminal,
                            "mean_photon": 0.1 + 0.001 * position,
                            "level_g": 0.8,
                            "level_e": 0.15,
                            "level_f": 0.05,
                            "logical_survival": 0.9,
                            "density_quantization_trace_distance_bound": 1e-7,
                        }
                    )
                    if terminal:
                        densities[row_id] = np.eye(3 * cutoff) / (3 * cutoff)
    config = {
        "cutoffs": [2, 3],
        "scenario_names": ["step"],
        "logical_state_schedule": list(states),
        "stage_partition": {"step": {"all": list(range(12))}},
        "diagnostic_contract": {
            "confidence": 0.95,
            "multiplier_replicates": 1999,
            "multiplier_seed_namespace": 1320000,
            "calibration_factor_source": {"required_factor": 1.0},
            "margins": {
                "ab_terminal_density_trace_distance": 0.15,
                "ab_terminal_mean_photon_difference": 0.08,
                "ab_terminal_level_probability_l1": 0.1,
                "ab_terminal_logical_survival_difference": 0.1,
                "cutoff_terminal_density_trace_distance": 0.1,
                "cutoff_terminal_mean_photon_difference": 0.08,
                "cutoff_terminal_level_probability_l1": 0.1,
                "cutoff_terminal_logical_survival_difference": 0.1,
            },
        },
    }
    return config, rows, densities


def test_diagnostic_enforces_state_stage_backend_and_cutoff_iut(monkeypatch):
    config, rows, densities = _fixture()
    monkeypatch.setattr(subject, "paired_density_trace_ucb", _fake_ucb)
    monkeypatch.setattr(subject, "paired_vector_norm_ucb", _fake_ucb)
    results = subject.evaluate_diagnostics(config, rows, densities)
    assert len(results) == 96
    assert len({row["gate_id"] for row in results}) == 96
    assert {row["logical_state"] for row in results} == {
        "0",
        "1",
        "+",
        "-",
        "+i",
        "-i",
    }
    assert {row["contrast"] for row in results} == {
        "same_cutoff_ab",
        "within_backend_cutoff",
    }
    assert all(row["cluster_count"] == 12 for row in results)
    assert all(row["design_pilot_only"] is True for row in results)
    assert all(row["signal_class"] == subject.INCONCLUSIVE_VERDICT for row in results)
    assert all(row["negative_interpretation"] == "inconclusive" for row in results)
    assert all(row["qualification_effect"] is None for row in results)
    assert all("pilot_pass" not in row for row in results)


def test_repeated_candidate_signal_is_promoted_but_never_qualified() -> None:
    rows = [
        {
            "signal_class": "CANDIDATE_EXPLORATORY_RISK_SIGNAL",
            "signal_repeated": False,
            "scenario": "step",
            "logical_state": "0",
            "stage": "terminal",
            "metric": "density_trace_distance",
            "contrast": "within_backend_cutoff",
            "cutoff_or_increment": "16->20",
            "backend_or_pair": backend,
            "qualification_effect": None,
        }
        for backend in ("A", "B")
    ]

    subject._promote_repeated_risk_signals(rows)

    assert all(row["signal_class"] == "STRONG_EXPLORATORY_RISK_SIGNAL" for row in rows)
    assert all(row["signal_repeated"] is True for row in rows)
    assert all(row["qualification_effect"] is None for row in rows)


def test_candidate_repeated_with_preexisting_strong_signal_is_promoted() -> None:
    rows = [
        {
            "signal_class": signal_class,
            "signal_repeated": False,
            "scenario": "step",
            "logical_state": "0",
            "stage": "terminal",
            "metric": "density_trace_distance",
            "contrast": "within_backend_cutoff",
            "cutoff_or_increment": "16->20",
            "backend_or_pair": backend,
            "qualification_effect": None,
        }
        for backend, signal_class in (
            ("A", "CANDIDATE_EXPLORATORY_RISK_SIGNAL"),
            ("B", "STRONG_EXPLORATORY_RISK_SIGNAL"),
        )
    ]

    subject._promote_repeated_risk_signals(rows)

    assert rows[0]["signal_class"] == "STRONG_EXPLORATORY_RISK_SIGNAL"
    assert rows[0]["signal_repeated"] is True
    assert rows[1]["signal_repeated"] is True
    assert rows[0]["qualification_effect"] is None


def test_missing_state_cluster_fails_closed(monkeypatch):
    config, rows, densities = _fixture()
    removed = rows.pop()
    densities.pop(removed["row_id"], None)
    monkeypatch.setattr(subject, "paired_density_trace_ucb", _fake_ucb)
    monkeypatch.setattr(subject, "paired_vector_norm_ucb", _fake_ucb)
    with pytest.raises(ValueError, match="round coverage"):
        subject.evaluate_diagnostics(config, rows, densities)


def test_cutoff_embedding_preserves_trace_and_top_left_block() -> None:
    lower = np.eye(6) / 6
    embedded = subject._embed_density(lower, 2, 3)
    assert embedded.shape == (9, 9)
    assert np.trace(embedded) == pytest.approx(1.0)
    assert np.array_equal(embedded[:6, :6], lower)
    assert np.count_nonzero(embedded[6:, :]) == 0


def test_multiplier_namespace_is_disjoint_and_stable() -> None:
    first = subject._seed(1320000, "gate-a")
    repeat = subject._seed(1320000, "gate-a")
    other = subject._seed(1320000, "gate-b")
    assert first == repeat
    assert first != other
    assert first >> 64 == 1320000


def test_load_inputs_rejects_rehashed_uq_claim_upgrade(tmp_path, monkeypatch) -> None:
    config = {
        "artifact_paths": {
            "run_identity": "run_identity.json",
            "execution_manifest": "manifest.json",
        },
        "diagnostic_contract": {
            "calibration_factor_source": {
                "required_analysis_sha256": "",
                "required_factor": 1.0,
                "required_coverage_all_passed": True,
            }
        },
    }
    run_identity = {
        "fixture": True,
        "input_snapshot": {
            "pilot_source": {
                "path": "pilot.py",
                "bytes": 1,
                "sha256": "x",
            }
        },
    }
    run_identity["analysis_sha256"] = subject._sha(run_identity)
    manifest = {
        "task_id": subject.TASK_ID,
        "schema_version": subject.pilot_runner.MANIFEST_SCHEMA,
        "status": subject.pilot_runner.STATUS,
        "scientific_verdict": None,
        "qualified_claim": None,
        "exception_rows": 0,
        "conservation_failure_rows": 0,
        "observed_cells": 32,
        "observed_rows": 27648,
        "claim_state": dict(subject.pilot_runner.CLAIM_BOUNDARY),
    }
    manifest["analysis_sha256"] = subject._sha(manifest)
    uq = {
        "selected_calibration_factor": 1.0,
        "validation_coverage_summary": {"all_cells_passed": True},
        "claim_state": {
            **subject.UQ_CLAIM_BOUNDARY,
            "external_sota": True,
        },
        "bindings": {"fixture": {"path": "fixture", "bytes": 0, "sha256": "x"}},
    }
    uq["analysis_sha256"] = subject._sha(uq)
    config["diagnostic_contract"]["calibration_factor_source"][
        "required_analysis_sha256"
    ] = uq["analysis_sha256"]
    extension = {
        "verdict": "PASS_PAIRED_CLUSTER_UQ_POWER_EXTENSION",
        "selected_formal_clusters_per_state": 384,
        "parent_analysis_sha256": uq["analysis_sha256"],
        "claim_state": dict(subject.EXTENSION_CLAIM_BOUNDARY),
        "bindings": {"fixture": {"path": "fixture", "bytes": 0, "sha256": "x"}},
    }
    extension["analysis_sha256"] = subject._sha(extension)
    (tmp_path / "run_identity.json").write_text(
        json.dumps(run_identity), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "uq.json").write_text(json.dumps(uq), encoding="utf-8")
    (tmp_path / "extension.json").write_text(json.dumps(extension), encoding="utf-8")
    monkeypatch.setattr(subject, "CONFIG_PATH", "config.json")
    monkeypatch.setattr(subject, "UQ_REPORT_PATH", "uq.json")
    monkeypatch.setattr(subject, "UQ_EXTENSION_PATH", "extension.json")
    monkeypatch.setattr(
        subject,
        "_activate_verified_diagnostic_modules",
        lambda *_args: None,
    )
    monkeypatch.setattr(subject.pilot_runner, "CONFIG_PATH", "config.json")
    monkeypatch.setattr(
        subject.pilot_runner,
        "load_pilot_config",
        lambda _root, **_kwargs: (config, {}),
    )
    monkeypatch.setattr(
        subject.pilot_runner,
        "materialize_execution_config",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        subject.pilot_runner,
        "_assert_input_snapshot",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        subject.pilot_runner,
        "_activate_verified_execution_modules",
        lambda *_args: None,
    )
    monkeypatch.setattr(subject.pilot_runner, "build_pilot_cells", lambda *_args: [])
    monkeypatch.setattr(subject.pilot_runner, "_verify_manifest", lambda *_args: None)
    monkeypatch.setattr(
        subject.pilot_runner,
        "_verify_complete_marker",
        lambda *_args: None,
    )
    monkeypatch.setattr(subject, "_verify_report_bindings", lambda *_args: None)

    with pytest.raises(ValueError, match="factor binding drift"):
        subject._load_inputs(tmp_path)


def test_report_transaction_commits_source_before_report(tmp_path, monkeypatch) -> None:
    report = {"bindings": {}}
    rows = [{"gate_id": "g", "pilot_pass": True}]
    monkeypatch.setattr(subject, "_build_report_core", lambda _root: (report, rows))
    monkeypatch.setattr(subject, "SOURCE_PATH", "source.csv")
    monkeypatch.setattr(subject, "REPORT_PATH", "report.json")
    original_atomic = subject._atomic_text

    def fail_report(path: Path, value: str) -> None:
        if path.name == "report.json":
            raise OSError("injected report commit failure")
        original_atomic(path, value)

    monkeypatch.setattr(subject, "_atomic_text", fail_report)
    with pytest.raises(OSError, match="injected"):
        subject.write_artifacts(tmp_path)
    assert (tmp_path / "source.csv").is_file()
    assert not (tmp_path / "report.json").exists()


def test_live_rebuild_failure_removes_published_report_and_completion(
    tmp_path, monkeypatch
) -> None:
    report = {
        "bindings": {},
        "input_snapshot_analysis_sha256": "snapshot",
    }
    rows = [{"gate_id": "g", "signal_class": subject.INCONCLUSIVE_VERDICT}]
    monkeypatch.setattr(subject, "_build_report_core", lambda _root: (report, rows))
    monkeypatch.setattr(subject, "SOURCE_PATH", "source.csv")
    monkeypatch.setattr(subject, "REPORT_PATH", "report.json")
    monkeypatch.setattr(subject, "COMPLETION_PATH", "completion.json")
    monkeypatch.setattr(
        subject,
        "DIAGNOSTIC_LOCK_PATH",
        "run/diagnostic.owner.lock",
    )
    monkeypatch.setattr(
        subject,
        "_finalized_report_document",
        lambda _root: (_ for _ in ()).throw(
            RuntimeError("injected live rebuild failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected live rebuild failure"):
        subject.write_artifacts(tmp_path)

    assert (tmp_path / "source.csv").is_file()
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "completion.json").exists()
    assert not (tmp_path / "run/diagnostic.owner.lock").exists()


def test_completion_rejects_report_or_source_rebound_to_attacker_content(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(subject, "REPORT_PATH", "report.json")
    monkeypatch.setattr(subject, "SOURCE_PATH", "source.csv")
    monkeypatch.setattr(subject, "COMPLETION_PATH", "completion.json")
    rows = [{"gate_id": "g", "signal_class": subject.INCONCLUSIVE_VERDICT}]
    (tmp_path / "source.csv").write_bytes(subject._source_csv_bytes(rows))
    report = {
        "schema_version": subject.SCHEMA,
        "status": subject.STATUS,
        "scientific_verdict": subject.INCONCLUSIVE_VERDICT,
        "qualified_claim": None,
        "claim_state": {
            "design_pilot_only": True,
            "external_sota": None,
            "official_puviani_exact": None,
        },
        "input_snapshot_analysis_sha256": "snapshot",
        "bindings": {
            "pilot_manifest": {
                "path": "manifest.json",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "source_data": subject._binding(tmp_path / "source.csv", tmp_path),
        },
        "transaction": subject._transaction_contract(),
    }
    report["analysis_sha256"] = subject._sha(report)
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (tmp_path / "report.json").write_text(report_text, encoding="utf-8")

    def commit_completion() -> None:
        completion = subject._completion_payload(tmp_path, report)
        (tmp_path / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    commit_completion()
    subject._verify_diagnostic_completion(tmp_path, report, rows)

    (tmp_path / "report.json").write_bytes(b"attacker-controlled non-json")
    commit_completion()
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        subject._verify_diagnostic_completion(tmp_path, report, rows)

    (tmp_path / "report.json").write_text(report_text, encoding="utf-8")
    (tmp_path / "source.csv").write_bytes(b"attacker-source")
    commit_completion()
    with pytest.raises(RuntimeError, match="source binding drift"):
        subject._verify_diagnostic_completion(tmp_path, report, rows)


def test_report_is_narrow_unpowered_nonranking_risk_signal(
    tmp_path, monkeypatch
) -> None:
    config = {
        "claim_boundary": dict(subject.pilot_runner.CLAIM_BOUNDARY),
        "artifact_paths": {"execution_manifest": "manifest.json"},
        "hardened_confirmation_source": {
            "report": {"path": "hardened.json"},
            "source_data": {"path": "hardened.csv"},
        },
    }
    diagnostics = [
        {
            "gate_id": "cutoff/24-28/A/step/0/terminal_density",
            "contrast": "within_backend_cutoff",
            "metric": "density_trace_distance",
            "cutoff_or_increment": "24->28",
            "estimate": 0.11,
            "upper_bound": 0.13,
            "margin": 0.1,
            "signal_class": "CANDIDATE_EXPLORATORY_RISK_SIGNAL",
        }
    ]
    hardened = {
        "frozen_parent_calibration_factor": 1.0,
        "analysis_sha256": subject.pilot_runner.HARDENED_CONFIRMATION_ANALYSIS_SHA256,
    }
    monkeypatch.setattr(
        subject,
        "_load_inputs",
        lambda _root: (
            config,
            {},
            {},
            {},
            hardened,
            {"pilot_source": {"fixture": True}},
        ),
    )
    monkeypatch.setattr(subject, "load_pilot_evidence", lambda *_args: ([], {}))
    monkeypatch.setattr(subject, "evaluate_diagnostics", lambda *_args: diagnostics)
    monkeypatch.setattr(
        subject.pilot_runner,
        "_assert_input_snapshot",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        subject,
        "_assert_diagnostic_import_snapshot",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda path, _root: {
            "path": Path(path).name,
            "bytes": 1,
            "sha256": "bound",
        },
    )

    report, _rows = subject._build_report_core(tmp_path)

    assert report["scientific_verdict"] == subject.RISK_VERDICT
    assert report["qualified_claim"] is None
    assert report["formal_design"] is None
    assert report["exploratory_signal_summary"]["global_iut_pass"] is None
    assert report["scope_guard"]["n12_role"] == "localization_only"
    assert report["scope_guard"]["equivalence_conclusion"] is None
    assert report["scope_guard"]["formal_cutoff_selection"] is None
    assert report["scope_guard"]["formal_sample_count_selection"] is None
    assert report["cutoff_32_exploratory_candidate"]["selected"] is None
    assert set(report["preserved_no_go"].values()) == {"NO_GO_PRESERVED"}
    assert report["downstream_state"] == {
        task: "BLOCKED"
        for task in (
            "T9.2.5",
            "T9.2.7",
            "T9.3.1",
            "T9.3.4",
            "T9.6.2",
            "T9.6.5",
        )
    }
    assert all(
        value is None
        for key, value in report["claim_state"].items()
        if key != "design_pilot_only"
    )

    diagnostics[0]["estimate"] = 0.01
    diagnostics[0]["upper_bound"] = 0.03
    diagnostics[0]["signal_class"] = subject.INCONCLUSIVE_VERDICT
    inconclusive, _rows = subject._build_report_core(tmp_path)
    assert inconclusive["scientific_verdict"] == subject.INCONCLUSIVE_VERDICT
    assert inconclusive["qualified_claim"] is None
    assert (
        inconclusive["scope_guard"]["absence_effect"]
        == "inconclusive; cannot qualify any candidate"
    )
    assert inconclusive["formal_design"] is None


def test_main_classifies_missing_or_exception_evidence_as_incomplete(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        subject,
        "write_artifacts",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing chunk")),
    )

    assert subject.main([]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "error_type": "FileNotFoundError",
        "qualified_claim": None,
        "scientific_verdict": subject.INCOMPLETE_VERDICT,
    }
