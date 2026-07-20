from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from cnn_fpga.benchmark import gqf_official_intake as intake


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_8_3_gqf_official_intake.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_current_intake_report_recomputes_and_all_gates_pass() -> None:
    report = _report()
    intake.verify_report(report)
    assert report["task_id"] == "T6.8.3"
    assert report["gate_summary"] == {"passed": 12, "failed": 0}
    assert all(report["gates"].values())
    assert report["verdict"].startswith("PASS_GQF_OFFICIAL_INTAKE_")


def test_official_checkout_is_pinned_pristine_and_content_bound() -> None:
    report = _report()
    upstream = report["upstream"]
    assert upstream["url"] == intake.UPSTREAM_URL
    assert upstream["commit"] == intake.UPSTREAM_COMMIT
    assert upstream["license"] == "MIT"
    assert upstream["tracked_status_clean"] is True
    assert len(upstream["tracked_files"]) == 12
    assert intake._tree_sha256(intake.UPSTREAM_ROOT, intake._tracked_files()) == upstream["tracked_tree_sha256"]
    status = subprocess.run(
        ["git", "-C", str(intake.UPSTREAM_ROOT), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout.strip() == ""


def test_patch_series_is_separate_effective_and_compiles() -> None:
    report = _report()
    patch = report["patch_manifest"]
    assert patch["worktree_method"] == "local-git-clone-v2"
    assert patch["upstream_checkout_modified"] is False
    assert patch["patched_tracked_tree_sha256"] != patch["upstream_tracked_tree_sha256"]
    assert patch["changed_tracked_files"] == [
        "GQF/GKP_environment.py",
        "GQF/feedback_GRAPE.py",
        "GQF/mesolve.py",
        "GQF/runner.py",
    ]
    assert len(patch["patches"]) == 4
    assert report["syntax_audit"]["upstream_failures"] == [
        {"path": "GQF/mesolve.py", "line": 13, "error": "IndentationError"}
    ]
    assert report["syntax_audit"]["patched_failure_count"] == 0
    for item in patch["patches"]:
        patch_path = ROOT / item["path"]
        check = subprocess.run(
            ["git", "-C", str(intake.UPSTREAM_ROOT), "apply", "--check", str(patch_path)],
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stderr


def test_environment_is_reconstructable_and_cpu_smoke_is_physical() -> None:
    report = _report()
    packages = report["environment"]["selected_packages"]
    assert packages["python"]["version"] == "3.9.18"
    assert packages["tensorflow"]["version"] == "2.10.1"
    assert packages["numpy"]["version"] == "1.23.5"
    assert packages["cudatoolkit"]["version"] == "11.2.2"
    assert packages["cudnn"]["version"] == "8.1.0.77"
    for lock in ("conda_explicit", "pip_freeze", "history"):
        bound = report["environment"]["locks"][lock]
        assert (ROOT / bound["path"]).is_file()
        assert len(bound["sha256"]) == 64

    cpu = report["smoke"]["cpu"]
    assert cpu["status"] == "PASS_CPU_FULL"
    assert cpu["environment_steps"] == 1
    assert cpu["cutoff"] == 8
    assert abs(cpu["gkp_state_norm"] - 1.0) <= 2.0e-5
    assert abs(cpu["rho_trace_real"] - 1.0) <= 2.0e-5
    assert abs(cpu["rho_trace_imag"]) <= 2.0e-5
    assert cpu["rho_hermitian_residual"] <= 2.0e-5
    assert cpu["rho_minimum_eigenvalue"] >= -2.0e-5
    assert 0.0 <= cpu["probability"] <= 1.0
    assert all(cpu["checks"].values())


def test_gpu_probe_is_either_real_qualified_artifact_or_honest_failure() -> None:
    gpu = _report()["smoke"]["gpu"]
    if gpu["status"] == "QUALIFIED_GPU_STATE":
        assert gpu["returncode"] == 0
        assert gpu["artifact"] is not None
    else:
        assert gpu["status"].startswith("UNQUALIFIED_")
        assert gpu["returncode"] != 0
        assert gpu["artifact"] is None
        assert gpu["stderr"]["contains_cusolver_failure"] is True
        stderr = (ROOT / gpu["stderr"]["path"]).read_text(encoding="utf-8", errors="replace")
        assert "cuSolver" in stderr or "cusolver" in stderr.lower()


def test_intake_cannot_be_promoted_to_exact_or_surpass_claim() -> None:
    report = _report()
    assert report["runner_manifest"]["scope"] == {
        "official_source_imported": True,
        "minimum_real_smoke": True,
        "paper_exact_reproduction": False,
        "trained_checkpoints_present": False,
        "claim": "INTAKE_ONLY_NOT_PAPER_REPRODUCTION",
    }
    assert report["claim_boundary"]["paper_exact_reproduction"] == "NOT_STARTED"
    assert report["claim_boundary"]["surpass_puviani_nmf"] == "PROHIBITED"

    forged = deepcopy(report)
    forged["runner_manifest"]["scope"]["paper_exact_reproduction"] = True
    assert intake.evaluate_gates(forged)["G10_runner_manifest_is_intake_only_and_checkpoint_honest"] is False
    with pytest.raises(ValueError, match="gates/verdict"):
        intake.verify_report(forged)


def test_hash_and_mutation_audits_fail_closed() -> None:
    report = _report()
    assert report["semantic_mutation_audit"]["count"] == 12
    assert report["semantic_mutation_audit"]["detected"] == 12
    assert {row["target_gate"] for row in report["semantic_mutation_audit"]["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"])

    forged = deepcopy(report)
    forged["bindings"]["cpu_smoke"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound artifact drifted"):
        intake.verify_report(forged)

