from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest

from cnn_fpga.benchmark import phase9_cutoff32_36_design_extension as subject
from cnn_fpga.benchmark import phase9_cutoff32_36_design_bootstrap as bootstrap
from cnn_fpga.benchmark import phase9_cutoff32_36_design_diagnostic as diagnostic
from cnn_fpga.benchmark import phase9_fresh_twin_qualification as runner


ROOT = Path(__file__).resolve().parents[1]


def _pending() -> dict:
    return json.loads(
        (ROOT / subject.PENDING_CONFIG_PATH).read_text(encoding="utf-8")
    )


def _interval(split: dict) -> set[int]:
    return set(range(int(split["start"]), int(split["start"]) + int(split["count"])))


def _valid_resource_preflight(config: dict) -> dict:
    contract = config["resource_preflight"]
    sample_count = 6 * int(contract["benchmark_trajectories_per_state"])
    records = []
    for identity in sorted(
        subject._expected_resource_preflight_identities(config),
        key=repr,
    ):
        cutoff, backend, layer, scenario, initial_state, action = identity
        records.append(
            {
                "cutoff": cutoff,
                "backend": backend,
                "layer": layer,
                "scenario": scenario,
                "initial_state": initial_state,
                "action": action,
                "sample_count": sample_count,
                "observed_rows": (
                    12 * sample_count if layer == "fault" else sample_count
                ),
                "terminal_density_count": sample_count,
                "elapsed_seconds": 1.0,
                "peak_process_rss_bytes": 100_000_000,
                "rss_delta_bytes": 1_000_000,
            }
        )
    report = {
        "benchmark_records": records,
        "baseline_rss_bytes": 99_000_000,
        "maximum_single_benchmark_rss_delta_bytes": 1_000_000,
        "minimum_accounted_per_worker_delta_bytes": int(
            contract["minimum_per_worker_delta_bytes"]
        ),
        "estimated_wall_seconds_with_safety_factor": 100.0,
        "estimated_total_rss_bytes": 1_000_000_000,
        "estimated_artifact_bytes": 1_000_000_000,
        "wall_limit_seconds": int(contract["maximum_estimated_wall_seconds"]),
        "rss_limit_bytes": int(contract["maximum_estimated_total_rss_bytes"]),
        "artifact_limit_bytes": int(contract["maximum_estimated_artifact_bytes"]),
        "configured_max_workers": int(config["max_workers"]),
        "seed_splits": deepcopy(contract["seed_splits"]),
        "design_outcomes_accessed": False,
        "passed": True,
    }
    report["analysis_sha256"] = subject._sha(report)
    return report


def test_pending_contract_binds_existing_hardened_confirmation() -> None:
    config = _pending()
    report = subject._validate_hardened_confirmation(ROOT, config)
    assert report["verdict"] == subject.HARDENED_CONFIRMATION_PASS
    assert report["analysis_sha256"] == subject.HARDENED_CONFIRMATION_ANALYSIS_SHA256
    assert report["selected_formal_clusters_per_state"] == 384
    assert report["qualified_claim"] is None
    assert all(
        value is None
        for key, value in report["claim_state"].items()
        if key != "hardened_confirmation_only"
    )


def test_released_child_materializes_fully_pinned_parent_without_patch() -> None:
    config, base = subject.load_pilot_config(ROOT, require_hardened=True)
    assert dict(config) == _pending()
    assert config.release_lineage["authorization_state"] == (
        subject.NARROW_AUTHORIZATION_STATE
    )
    assert config.release_lineage["uq_preflights"] == (
        subject._uq_release_evidence(config)
    )
    assert base["task_id"] == "T-RISK-20260726-01"


def test_verified_bootstrap_binds_exact_release_runner_and_diagnostic() -> None:
    assert bootstrap.RELEASED_CHILD_BYTES == (
        ROOT / bootstrap.RELEASED_CHILD_PATH
    ).stat().st_size
    assert bootstrap.RELEASED_CHILD_SHA256 == sha256(
        (ROOT / bootstrap.RELEASED_CHILD_PATH).read_bytes()
    ).hexdigest()
    assert bootstrap.RUNNER_SHA256 == sha256(
        (ROOT / bootstrap.RUNNER_PATH).read_bytes()
    ).hexdigest()
    assert bootstrap.DIAGNOSTIC_SHA256 == sha256(
        (ROOT / bootstrap.DIAGNOSTIC_PATH).read_bytes()
    ).hexdigest()
    assert bootstrap.EXTERNAL_LAUNCHER_SHA256 == sha256(
        bootstrap.EXTERNAL_LAUNCHER_SOURCE.encode("utf-8")
    ).hexdigest()
    assert subject.EXTERNAL_LAUNCHER_SHA256 == (
        bootstrap.EXTERNAL_LAUNCHER_SHA256
    )
    assert diagnostic.EXTERNAL_LAUNCHER_SHA256 == (
        bootstrap.EXTERNAL_LAUNCHER_SHA256
    )


def test_byte_attested_paths_have_cross_platform_git_attributes() -> None:
    raw_paths = [
        "cnn_fpga/benchmark/phase9_cutoff32_36_design_bootstrap.py",
        "cnn_fpga/benchmark/phase9_cutoff32_36_design_extension.py",
        "cnn_fpga/benchmark/phase9_cutoff32_36_design_diagnostic.py",
        "physics/phase9_high_cutoff_runtime_adapter.py",
        subject.PENDING_CONFIG_PATH,
        subject.CONFIG_PATH,
        subject.RELEASE_RECEIPT_PATH,
        "docs/t_risk_20260727_01_uq_hardened_confirmation.json",
        "docs/t_risk_20260728_01_density_uq_preflight.json",
        "docs/t_risk_20260728_02_scalar_uq_calibration.json",
        (
            "runs/t_risk_20260728_01_density_uq_preflight/chunks/"
            "heavy_tail_rare_coherent__d84__n12__delta0p0.json"
        ),
        (
            "runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3/"
            "chunks/pilot_c28_fault_burst_A__1403be227f3fd32b.csv"
        ),
    ]
    for path in raw_paths:
        attributes = subprocess.check_output(
            ["git", "check-attr", "text", "eol", "filter", "--", path],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        )
        assert f"{path}: text: unset" in attributes
        assert f"{path}: filter: unspecified" in attributes

    binary_paths = {
        "docs/t_risk_20260728_02_scalar_uq_selection_a.csv": "lfs",
        (
            "runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3/"
            "chunks/pilot_c28_fault_burst_A__1403be227f3fd32b.npz"
        ): "lfs",
    }
    for path, expected_filter in binary_paths.items():
        attributes = subprocess.check_output(
            ["git", "check-attr", "text", "filter", "--", path],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        )
        assert f"{path}: text: unset" in attributes
        assert f"{path}: filter: {expected_filter}" in attributes


def test_direct_runner_and_diagnostic_imports_are_not_executable() -> None:
    with pytest.raises(RuntimeError, match="trusted-operator bootstrap"):
        subject._require_verified_self_import()
    with pytest.raises(RuntimeError, match="trusted-operator bootstrap"):
        diagnostic._require_verified_self_import()


def test_isolated_verified_probe_attests_six_thread_workers() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            bootstrap.EXTERNAL_LAUNCHER_SOURCE,
            str(ROOT),
            sha256(
                (
                    ROOT
                    / "cnn_fpga/benchmark/phase9_cutoff32_36_design_bootstrap.py"
                ).read_bytes()
            ).hexdigest(),
            bootstrap.EXTERNAL_LAUNCHER_SHA256,
            "probe",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["worker_attestations"] == 6
    assert payload["runner_sha256"] == bootstrap.RUNNER_SHA256
    assert (
        payload["high_cutoff_adapter_receipt"][
            "extended_max_supported_cutoff"
        ]
        == 36
    )


def test_pending_contract_binds_density_and_independently_verified_scalar_uq() -> None:
    bindings = subject._validate_uq_preflight_sources(ROOT, _pending())
    assert set(bindings) == {
        "density_uq_report",
        "density_uq_source_data",
        "scalar_uq_report",
        "scalar_uq_independent_verification",
        "scalar_uq_selection_a_source_data",
        "scalar_uq_selection_b_source_data",
        "scalar_uq_confirmation_source_data",
    }
    assert (
        bindings["scalar_uq_report"]["path"]
        == "docs/t_risk_20260728_02_scalar_uq_calibration.json"
    )
    assert (
        bindings["scalar_uq_independent_verification"]["path"]
        == "docs/t_risk_20260728_02_scalar_uq_calibration_verification.json"
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("scalar", "required_analysis_sha256"),
            "0" * 64,
            "identity drift",
        ),
        (
            ("scalar", "required_qualified_claim"),
            None,
            "lane contract drift",
        ),
        (
            ("density", "report", "bytes"),
            1,
            "binding",
        ),
    ],
)
def test_uq_release_mutations_fail_closed(path, value, message) -> None:
    config = _pending()
    target = config["uq_preflight_sources"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match=message):
        subject._validate_uq_preflight_sources(ROOT, config)


def test_failed_scalar_v1_cannot_be_substituted_for_calibrated_v2() -> None:
    config = _pending()
    failed_path = ROOT / "docs/t_risk_20260728_01_scalar_uq_preflight.json"
    failed_binding = {
        "path": failed_path.relative_to(ROOT).as_posix(),
        "bytes": failed_path.stat().st_size,
        "sha256": sha256(failed_path.read_bytes()).hexdigest(),
    }
    config["uq_preflight_sources"]["scalar"]["report"] = failed_binding
    with pytest.raises(RuntimeError, match="identity drift"):
        subject._validate_uq_preflight_sources(ROOT, config)


def test_design_resource_and_powered_splits_are_pairwise_disjoint() -> None:
    config = _pending()
    design = config["seed_splits"]
    resource = config["resource_preflight"]["seed_splits"]
    formal = config["powered_formal_preregistration"]["seed_splits"]
    intervals = []
    for registry in (design, resource, formal):
        for name in (
            "round_backend_a",
            "round_backend_b",
            "trajectory_backend_a",
            "trajectory_backend_b",
            "heldout_common",
        ):
            intervals.append((registry is resource, name, _interval(registry[name])))
    for index, (_, _, left) in enumerate(intervals):
        for _, _, right in intervals[index + 1 :]:
            assert not left & right
    assert resource["disjoint_from_design_and_powered_formal"] is True
    assert not any(
        int(
            config["powered_formal_preregistration"][
                "multiplier_seed_namespace"
            ]
        )
        in interval
        for _, _, interval in intervals
    )


def test_resource_preflight_requires_exact_22_cell_identity_set() -> None:
    config = _pending()
    report = _valid_resource_preflight(config)
    subject._validate_resource_preflight(config, report)

    duplicate = deepcopy(report)
    duplicate["benchmark_records"][-1] = deepcopy(
        duplicate["benchmark_records"][0]
    )
    duplicate["analysis_sha256"] = subject._sha(
        {
            key: value
            for key, value in duplicate.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="duplicate resource preflight identity"):
        subject._validate_resource_preflight(config, duplicate)


def test_resource_preflight_is_published_before_design_chunks() -> None:
    config = _pending()
    config["artifact_paths"]["run_directory"] = "run"
    report = _valid_resource_preflight(config)

    with TemporaryDirectory(dir=ROOT) as directory:
        root = Path(directory)
        live = subject._publish_resource_preflight(root, config, report)
        path = root / "run/resource_preflight.json"
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8")) == live
        subject._validate_resource_preflight(config, live)


def test_failed_resource_limit_evidence_remains_published() -> None:
    config = _pending()
    config["artifact_paths"]["run_directory"] = "run"
    report = _valid_resource_preflight(config)
    report["estimated_wall_seconds_with_safety_factor"] = (
        int(config["resource_preflight"]["maximum_estimated_wall_seconds"]) + 1
    )
    report["passed"] = False
    report["analysis_sha256"] = subject._sha(
        {
            key: value
            for key, value in report.items()
            if key != "analysis_sha256"
        }
    )

    with TemporaryDirectory(dir=ROOT) as directory:
        root = Path(directory)
        with pytest.raises(
            RuntimeError,
            match="cutoff32/36 resource preflight failed",
        ):
            subject._publish_resource_preflight(root, config, report)
        path = root / "run/resource_preflight.json"
        archived = json.loads(path.read_text(encoding="utf-8"))
        assert archived["passed"] is False
        assert archived["analysis_sha256"] == report["analysis_sha256"]


def test_design_matrix_is_complete_and_not_a_single_cell_demo(monkeypatch) -> None:
    config = _pending()
    base = json.loads(
        (ROOT / config["base_config"]["path"]).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(subject, "runner", runner)
    execution = subject.materialize_execution_config(config, base)
    cells = subject.build_pilot_cells(config, execution)

    assert len(cells) == 22
    assert sum(cell.layer == "fault" for cell in cells) == 16
    assert sum(cell.layer == "shared" for cell in cells) == 6
    assert sum(cell.expected_rows for cell in cells) == 14256
    assert {cell.cutoff for cell in cells if cell.layer == "fault"} == {32, 36}
    assert {
        (cell.cutoff, cell.backend)
        for cell in cells
        if cell.layer == "shared"
    } == {
        (cutoff, backend)
        for cutoff in (28, 32, 36)
        for backend in ("A", "B")
    }
    assert {
        cell.scenario for cell in cells if cell.layer == "fault"
    } == {"step", "telegraph", "burst", "compound"}
    assert all(cell.sample_count == 72 for cell in cells)
    assert len({cell.chunk_id for cell in cells}) == 22
    fault = next(cell for cell in cells if cell.layer == "fault")
    shared = next(cell for cell in cells if cell.layer == "shared")
    assert runner._seed_for(execution, fault, 0) in {1430000, 1431000}
    assert runner._seed_for(execution, shared, 0) in {1433000, 1434000}


def test_source_and_reference_bindings_match_live_bytes() -> None:
    config = _pending()
    for binding in config["source_bindings"].values():
        path = ROOT / binding["path"]
        assert sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    base = ROOT / config["base_config"]["path"]
    assert sha256(base.read_bytes()).hexdigest() == config["base_config"]["sha256"]
    reference = subject._reference_input_bindings(ROOT, config)
    assert sum(name.startswith("cutoff28_receipt_") for name in reference) == 8
    assert sum(name.startswith("cutoff28_csv_") for name in reference) == 8
    assert sum(name.startswith("cutoff28_npz_") for name in reference) == 8
    assert (
        config["source_bindings"]["high_cutoff_adapter"]["path"]
        == "physics/phase9_high_cutoff_runtime_adapter.py"
    )


def test_powered_formal_contract_cannot_claim_external_or_physical_results() -> None:
    config = _pending()
    formal = config["powered_formal_preregistration"]
    assert formal["cutoffs"] == [28, 32, 36]
    assert formal["fresh_rerun_all_cutoffs"] is True
    assert formal["clusters_per_state"] == 384
    assert formal["trajectory_count"] == 2304
    assert formal["seed_splits"]["round_backend_a"]["count"] == 2304
    assert formal["seed_splits"]["round_backend_b"]["count"] == 2304
    assert formal["prior_passing_gate_composition"]["allowed"] is False
    assert (
        formal["full_fresh_qualification_required"]
        == {
            "required": True,
            "same_runtime_source_set_for_every_gate": True,
            "all_previous_gate_families_rerun": True,
            "all_high_cutoff_design_gates_rerun_powered": True,
            "old_1562_passing_gates_vote": False,
            "old_27_failing_gates_vote": False,
            "independent_final_verifier_required": True,
            "failure_effect": "NO_GO_TWIN_QUALIFICATION_NO_SEED_EXTENSION",
        }
    )
    assert formal["design_outcome_may_not_change_contract"] is True
    assert formal["official_puviani_sota_claims"] is None
    assert all(
        value is None
        for key, value in config["claim_boundary"].items()
        if key != "design_extension_only"
    )
