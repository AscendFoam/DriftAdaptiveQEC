"""T6.8.3 official GQF source, environment, patch, and smoke intake.

The upstream checkout remains pristine.  Compatibility patches are applied to
a content-addressed derived worktree, and all executable evidence is bound to
the upstream commit, tracked-tree hash, patch hashes, environment locks, and
the project-side smoke implementation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.8.3"
SCHEMA_VERSION = "t6.8.3-gqf-official-intake-v1"
UPSTREAM_ROOT = ROOT / "third_party" / "GQF"
UPSTREAM_URL = "https://github.com/Matteo-Puviani/GQF.git"
UPSTREAM_COMMIT = "c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d"
PATCH_ROOT = ROOT / "third_party" / "patches" / "GQF" / UPSTREAM_COMMIT
ENV_NAME = "GQFEnv"
LOCK_ROOT = ROOT / "configs" / "gqf_official"
CONDA_EXPLICIT = LOCK_ROOT / "GQFEnv.conda-explicit-win-64.txt"
PIP_FREEZE = LOCK_ROOT / "GQFEnv.pip-freeze.txt"
ENV_HISTORY = LOCK_ROOT / "GQFEnv.history.yml"
RUNNER_MANIFEST = LOCK_ROOT / "runner_manifest.json"
PATCH_MANIFEST = LOCK_ROOT / "patch_manifest.json"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_8_3_gqf_official_intake.json"
CPU_SMOKE_ARTIFACT = ROOT / "docs" / "t6_8_3_gqf_cpu_smoke.json"
GPU_STDOUT = ROOT / "docs" / "t6_8_3_gqf_gpu_probe.stdout.log"
GPU_STDERR = ROOT / "docs" / "t6_8_3_gqf_gpu_probe.stderr.log"
SMOKE_SCRIPT = ROOT / "scripts" / "gqf_official_smoke.py"
WORKTREE_ROOT = ROOT / "runs" / "t6_8_3_gqf_worktrees"
WORKTREE_METHOD = "local-git-clone-v2"
EXPECTED_TOP_LEVEL_REQUIREMENTS = ("numpy", "matplotlib", "tensorflow", "gym")
EXPECTED_PATCHES = (
    "0001-fix-mesolve-init-docstring-indentation.patch",
    "0002-replace-string-identity-comparisons.patch",
    "0003-fix-official-runner-model-token.patch",
    "0004-fix-test-evaluation-zero-buffer-shape.patch",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(("git", "-C", str(UPSTREAM_ROOT), *args), check=check)


def _tracked_files() -> tuple[str, ...]:
    return tuple(
        line.strip().replace("\\", "/")
        for line in _git("ls-files").stdout.splitlines()
        if line.strip()
    )


def _tree_sha256(root: Path, relative_files: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in relative_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _patch_files() -> tuple[Path, ...]:
    series = tuple(
        line.strip()
        for line in (PATCH_ROOT / "series.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if series != EXPECTED_PATCHES:
        raise ValueError(f"GQF patch series drifted: {series}")
    files = tuple(PATCH_ROOT / name for name in series)
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("GQF patch series is incomplete")
    return files


def _patch_series_sha256(patches: Sequence[Path]) -> str:
    return _json_sha256(
        [{"name": path.name, "sha256": _sha256(path)} for path in patches]
    )


def _patched_worktree(
    tracked_files: Sequence[str], upstream_sha256: str, patches: Sequence[Path]
) -> tuple[Path, dict[str, Any]]:
    patch_sha256 = _patch_series_sha256(patches)
    worktree = WORKTREE_ROOT / f"{UPSTREAM_COMMIT[:12]}-{patch_sha256[:12]}-{WORKTREE_METHOD}"
    local_manifest = worktree / ".gqf_adapter_manifest.json"
    expected_prefix = {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tracked_tree_sha256": upstream_sha256,
        "patch_series_sha256": patch_sha256,
        "worktree_method": WORKTREE_METHOD,
        "patches": [
            {"path": _relative(path), "sha256": _sha256(path)} for path in patches
        ],
    }
    if not worktree.exists():
        WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
        _run(
            (
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(UPSTREAM_ROOT),
                str(worktree),
            )
        )
        _run(("git", "checkout", "--detach", UPSTREAM_COMMIT), cwd=worktree)
        for patch in patches:
            _run(("git", "apply", str(patch.resolve())), cwd=worktree)
        patched_sha256 = _tree_sha256(worktree, tracked_files)
        changed_tracked_files = sorted(
            line[3:].replace("\\", "/")
            for line in _run(
                ("git", "status", "--short", "--untracked-files=no"), cwd=worktree
            ).stdout.splitlines()
            if line.strip()
        )
        payload = {
            **expected_prefix,
            "patched_tracked_tree_sha256": patched_sha256,
            "changed_tracked_files": changed_tracked_files,
        }
        local_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    for key, value in expected_prefix.items():
        if payload.get(key) != value:
            raise ValueError(f"derived GQF worktree manifest drifted: {key}")
    if payload.get("patched_tracked_tree_sha256") != _tree_sha256(worktree, tracked_files):
        raise ValueError("derived GQF worktree source drifted")
    if payload.get("patched_tracked_tree_sha256") == upstream_sha256:
        raise ValueError("GQF patch application was a no-op")
    if payload.get("changed_tracked_files") != [
        "GQF/GKP_environment.py",
        "GQF/feedback_GRAPE.py",
        "GQF/mesolve.py",
        "GQF/runner.py",
    ]:
        raise ValueError("GQF patch application changed an unexpected file set")
    return worktree, payload


def _syntax_audit(root: Path, relative_files: Sequence[str], python: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in relative_files:
        if not relative.endswith(".py"):
            continue
        path = root / relative
        code = (
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1]); compile(p.read_text(encoding='utf-8'),str(p),'exec')"
        )
        completed = _run((str(python), "-c", code, str(path)), check=False)
        rows.append(
            {
                "path": relative,
                "passed": completed.returncode == 0,
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
        )
    return rows


def _conda_executable() -> str:
    executable = shutil.which("conda")
    if executable is None:
        raise FileNotFoundError("conda executable is unavailable")
    return executable


def _environment_snapshot() -> tuple[Path, list[dict[str, Any]]]:
    conda = _conda_executable()
    envs = json.loads(_run((conda, "env", "list", "--json")).stdout)["envs"]
    candidates = [Path(path) for path in envs if Path(path).name.lower() == ENV_NAME.lower()]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {ENV_NAME}, found {candidates}")
    env_root = candidates[0]
    python = env_root / ("python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError(python)

    explicit = _run((conda, "list", "-n", ENV_NAME, "--explicit")).stdout
    packages = json.loads(_run((conda, "list", "-n", ENV_NAME, "--json")).stdout)
    freeze = _run(
        (conda, "run", "-n", ENV_NAME, "--no-capture-output", "python", "-m", "pip", "freeze", "--all")
    ).stdout
    history = _run((conda, "env", "export", "-n", ENV_NAME, "--from-history")).stdout
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    CONDA_EXPLICIT.write_text(explicit, encoding="utf-8")
    PIP_FREEZE.write_text(freeze, encoding="utf-8")
    ENV_HISTORY.write_text(history, encoding="utf-8")
    return python, packages


def _package_table(packages: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    selected = {
        "python",
        "pip",
        "numpy",
        "tensorflow",
        "gym",
        "matplotlib",
        "cudatoolkit",
        "cudnn",
    }
    return {
        str(row["name"]): {
            key: row.get(key) for key in ("version", "build_string", "channel")
        }
        for row in packages
        if str(row.get("name")) in selected
    }


def _smoke_environment(device: str) -> dict[str, str]:
    env = dict(os.environ)
    env["TF_CPP_MIN_LOG_LEVEL"] = "1"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _run_smokes(worktree: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    conda = _conda_executable()
    CPU_SMOKE_ARTIFACT.unlink(missing_ok=True)
    cpu_command = (
        conda,
        "run",
        "-n",
        ENV_NAME,
        "--no-capture-output",
        "python",
        str(SMOKE_SCRIPT),
        "--gqf-root",
        str(worktree / "GQF"),
        "--mode",
        "cpu_full",
        "--artifact",
        str(CPU_SMOKE_ARTIFACT),
    )
    cpu_completed = _run(cpu_command, env=_smoke_environment("cpu"))
    cpu = json.loads(CPU_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
    cpu["command"] = list(cpu_command)
    cpu["stdout_sha256"] = hashlib.sha256(cpu_completed.stdout.encode("utf-8")).hexdigest()
    cpu["stderr_sha256"] = hashlib.sha256(cpu_completed.stderr.encode("utf-8")).hexdigest()

    gpu_artifact = ROOT / "docs" / "t6_8_3_gqf_gpu_state_smoke.json"
    gpu_artifact.unlink(missing_ok=True)
    gpu_command = (
        conda,
        "run",
        "-n",
        ENV_NAME,
        "--no-capture-output",
        "python",
        str(SMOKE_SCRIPT),
        "--gqf-root",
        str(worktree / "GQF"),
        "--mode",
        "gpu_state",
        "--artifact",
        str(gpu_artifact),
    )
    gpu_completed = _run(gpu_command, env=_smoke_environment("gpu"), check=False)
    GPU_STDOUT.write_text(gpu_completed.stdout, encoding="utf-8")
    GPU_STDERR.write_text(gpu_completed.stderr, encoding="utf-8")
    if gpu_completed.returncode == 0 and gpu_artifact.is_file():
        gpu_status = "QUALIFIED_GPU_STATE"
        gpu_payload = json.loads(gpu_artifact.read_text(encoding="utf-8"))
    elif "cuSolver" in gpu_completed.stderr or "cusolver" in gpu_completed.stderr:
        gpu_status = "UNQUALIFIED_CUSOLVER_FATAL"
        gpu_payload = None
    else:
        gpu_status = "UNQUALIFIED_OTHER_FAILURE"
        gpu_payload = None
    gpu = {
        "status": gpu_status,
        "returncode": gpu_completed.returncode,
        "command": list(gpu_command),
        "artifact": _relative(gpu_artifact) if gpu_artifact.is_file() else None,
        "artifact_payload": gpu_payload,
        "stdout": {
            "path": _relative(GPU_STDOUT),
            "sha256": _sha256(GPU_STDOUT),
        },
        "stderr": {
            "path": _relative(GPU_STDERR),
            "sha256": _sha256(GPU_STDERR),
            "contains_cusolver_failure": "cusolver" in gpu_completed.stderr.lower(),
        },
    }
    return cpu, gpu


def _write_manifests(
    tracked_files: Sequence[str],
    upstream_sha256: str,
    worktree: Path,
    local_patch_manifest: Mapping[str, Any],
    packages: Sequence[Mapping[str, Any]],
    gpu_status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requirements = tuple(
        line.strip()
        for line in (UPSTREAM_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    patch_manifest = {
        "schema_version": "t6.8.3-gqf-patch-manifest-v1",
        **dict(local_patch_manifest),
        "derived_worktree": _relative(worktree),
        "upstream_checkout_modified": bool(_git("status", "--porcelain", "--untracked-files=no").stdout.strip()),
    }
    runner_manifest = {
        "schema_version": "t6.8.3-gqf-runner-manifest-v1",
        "repository": UPSTREAM_URL,
        "commit": UPSTREAM_COMMIT,
        "tracked_tree_sha256": upstream_sha256,
        "tracked_files": list(tracked_files),
        "upstream_requirements_unpinned": list(requirements),
        "official_runner_defaults": {
            "source": "third_party/GQF/GQF/runner.py",
            "N": 100,
            "Delta": 0.34,
            "max_steps": 21,
            "batch_size": 8,
            "num_controls": 15,
            "num_training_episodes": 101,
            "num_evaluation_episodes": 1,
            "learning_rate": 1.0e-4,
            "dynamics_type": "Sivak2023",
            "noise_level": "high",
            "upstream_model_token": "RNN",
            "patched_model_token": "rNN",
        },
        "environment": {
            "name": ENV_NAME,
            "selected_packages": _package_table(packages),
            "conda_explicit": {"path": _relative(CONDA_EXPLICIT), "sha256": _sha256(CONDA_EXPLICIT)},
            "pip_freeze": {"path": _relative(PIP_FREEZE), "sha256": _sha256(PIP_FREEZE)},
            "history": {"path": _relative(ENV_HISTORY), "sha256": _sha256(ENV_HISTORY)},
            "gpu_qualification": gpu_status,
        },
        "adapter": {
            "patch_manifest": _relative(PATCH_MANIFEST),
            "derived_worktree": _relative(worktree),
            "smoke_script": _relative(SMOKE_SCRIPT),
            "smoke_script_sha256": _sha256(SMOKE_SCRIPT),
        },
        "scope": {
            "official_source_imported": True,
            "minimum_real_smoke": True,
            "paper_exact_reproduction": False,
            "trained_checkpoints_present": any(path.lower().endswith((".h5", ".keras", ".ckpt")) for path in tracked_files),
            "claim": "INTAKE_ONLY_NOT_PAPER_REPRODUCTION",
        },
    }
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    PATCH_MANIFEST.write_text(json.dumps(patch_manifest, indent=2) + "\n", encoding="utf-8")
    RUNNER_MANIFEST.write_text(json.dumps(runner_manifest, indent=2) + "\n", encoding="utf-8")
    return patch_manifest, runner_manifest


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    upstream = report["upstream"]
    environment = report["environment"]
    syntax = report["syntax_audit"]
    patch = report["patch_manifest"]
    cpu = report["smoke"]["cpu"]
    gpu = report["smoke"]["gpu"]
    runner = report["runner_manifest"]
    bindings = report["bindings"]
    package_versions = environment["selected_packages"]
    expected_gpu_honesty = (
        (gpu["status"] == "QUALIFIED_GPU_STATE" and gpu["returncode"] == 0 and gpu["artifact"] is not None)
        or (gpu["status"].startswith("UNQUALIFIED_") and gpu["returncode"] != 0 and gpu["artifact"] is None)
    )
    return {
        "G01_official_url_and_commit_are_pinned": upstream["url"] == UPSTREAM_URL and upstream["commit"] == UPSTREAM_COMMIT,
        "G02_mit_license_and_pristine_tracked_checkout_are_bound": upstream["license"] == "MIT" and upstream["tracked_status_clean"] is True,
        "G03_complete_tracked_tree_has_content_hash": len(upstream["tracked_files"]) >= 12 and len(upstream["tracked_tree_sha256"]) == 64,
        "G04_python_dependency_and_cuda_locks_are_complete": all(name in package_versions for name in ("python", "numpy", "tensorflow", "gym", "matplotlib", "cudatoolkit", "cudnn")) and all(item["sha256"] == bindings[item_id]["sha256"] for item_id, item in (("conda_explicit", environment["locks"]["conda_explicit"]), ("pip_freeze", environment["locks"]["pip_freeze"]), ("history", environment["locks"]["history"]))),
        "G05_pristine_syntax_blocker_is_disclosed_not_hidden": syntax["upstream_failures"] == [{"path": "GQF/mesolve.py", "line": 13, "error": "IndentationError"}],
        "G06_patch_series_is_hash_bound_effective_and_patched_tree_compiles": patch["upstream_checkout_modified"] is False and len(patch["patches"]) == 4 and patch["patched_tracked_tree_sha256"] != upstream["tracked_tree_sha256"] and patch["changed_tracked_files"] == ["GQF/GKP_environment.py", "GQF/feedback_GRAPE.py", "GQF/mesolve.py", "GQF/runner.py"] and syntax["patched_failure_count"] == 0,
        "G07_upstream_and_adapter_worktrees_are_separate": patch["derived_worktree"] != "third_party/GQF" and runner["adapter"]["derived_worktree"] == patch["derived_worktree"],
        "G08_cpu_smoke_executes_real_state_and_environment_step": cpu["status"] == "PASS_CPU_FULL" and cpu["environment_steps"] == 1 and abs(cpu["gkp_state_norm"] - 1.0) <= 2.0e-5 and all(cpu["checks"].values()),
        "G09_gpu_qualification_matches_isolated_probe": expected_gpu_honesty,
        "G10_runner_manifest_is_intake_only_and_checkpoint_honest": runner["scope"] == {"official_source_imported": True, "minimum_real_smoke": True, "paper_exact_reproduction": False, "trained_checkpoints_present": False, "claim": "INTAKE_ONLY_NOT_PAPER_REPRODUCTION"},
        "G11_all_live_files_and_manifests_are_hash_bound": all(len(item["sha256"]) == 64 for item in bindings.values()),
        "G12_target_specific_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 12,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    mutation_count = 12

    def attempt(name: str, target_gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {
            "count": mutation_count,
            "detected": mutation_count,
            "cases": [],
        }
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[target_gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": target_gate, "rejected": rejected})

    attempt("wrong_commit", "G01_official_url_and_commit_are_pinned", lambda x: x["upstream"].update(commit="0" * 40))
    attempt("hide_dirty_upstream", "G02_mit_license_and_pristine_tracked_checkout_are_bound", lambda x: x["upstream"].update(tracked_status_clean=False))
    attempt("truncate_tree", "G03_complete_tracked_tree_has_content_hash", lambda x: x["upstream"].update(tracked_files=["README.md"]))
    attempt("drop_cuda_lock", "G04_python_dependency_and_cuda_locks_are_complete", lambda x: x["environment"]["selected_packages"].pop("cudnn"))
    attempt("hide_upstream_syntax_error", "G05_pristine_syntax_blocker_is_disclosed_not_hidden", lambda x: x["syntax_audit"].update(upstream_failures=[]))
    attempt("drop_patch", "G06_patch_series_is_hash_bound_effective_and_patched_tree_compiles", lambda x: x["patch_manifest"].update(patches=x["patch_manifest"]["patches"][:3]))
    attempt("alias_upstream_adapter", "G07_upstream_and_adapter_worktrees_are_separate", lambda x: x["patch_manifest"].update(derived_worktree="third_party/GQF"))
    attempt("import_only_smoke", "G08_cpu_smoke_executes_real_state_and_environment_step", lambda x: x["smoke"]["cpu"].update(environment_steps=0))
    attempt("fake_gpu_pass", "G09_gpu_qualification_matches_isolated_probe", lambda x: x["smoke"]["gpu"].update(status="QUALIFIED_GPU_STATE"))
    attempt("fake_exact_reproduction", "G10_runner_manifest_is_intake_only_and_checkpoint_honest", lambda x: x["runner_manifest"]["scope"].update(paper_exact_reproduction=True))
    attempt("truncate_bound_hash", "G11_all_live_files_and_manifests_are_hash_bound", lambda x: x["bindings"]["cpu_smoke"].update(sha256="0"))
    attempt("forge_mutation_count", "G12_target_specific_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": mutation_count, "detected": mutation_count - 1, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD").stdout.strip()
    remote = _git("remote", "get-url", "origin").stdout.strip()
    if commit != UPSTREAM_COMMIT or remote != UPSTREAM_URL:
        raise ValueError(f"official GQF identity mismatch: {remote}@{commit}")
    tracked_files = _tracked_files()
    upstream_sha256 = _tree_sha256(UPSTREAM_ROOT, tracked_files)
    tracked_status = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    patches = _patch_files()
    worktree, local_patch_manifest = _patched_worktree(tracked_files, upstream_sha256, patches)
    env_python, packages = _environment_snapshot()
    upstream_syntax = _syntax_audit(UPSTREAM_ROOT, tracked_files, env_python)
    patched_syntax = _syntax_audit(worktree, tracked_files, env_python)
    upstream_failures = []
    for row in upstream_syntax:
        if row["passed"]:
            continue
        if row["path"] == "GQF/mesolve.py" and "IndentationError" in row["stderr"]:
            upstream_failures.append({"path": row["path"], "line": 13, "error": "IndentationError"})
        else:
            upstream_failures.append({"path": row["path"], "line": None, "error": row["stderr"]})
    cpu, gpu = _run_smokes(worktree)
    patch_manifest, runner_manifest = _write_manifests(
        tracked_files,
        upstream_sha256,
        worktree,
        local_patch_manifest,
        packages,
        gpu["status"],
    )
    selected_packages = _package_table(packages)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "url": remote,
            "commit": commit,
            "commit_date": _git("show", "-s", "--format=%aI", "HEAD").stdout.strip(),
            "tracked_files": list(tracked_files),
            "tracked_tree_sha256": upstream_sha256,
            "tracked_status_clean": not bool(tracked_status),
            "license": "MIT" if "MIT License" in (UPSTREAM_ROOT / "LICENSE").read_text(encoding="utf-8") else "UNKNOWN",
            "requirements": list(EXPECTED_TOP_LEVEL_REQUIREMENTS),
        },
        "environment": {
            "name": ENV_NAME,
            "python_executable": str(env_python),
            "selected_packages": selected_packages,
            "locks": runner_manifest["environment"] | {"gpu_qualification": gpu["status"]},
        },
        "syntax_audit": {
            "upstream": upstream_syntax,
            "upstream_failures": upstream_failures,
            "patched": patched_syntax,
            "patched_failure_count": sum(not row["passed"] for row in patched_syntax),
        },
        "patch_manifest": patch_manifest,
        "runner_manifest": runner_manifest,
        "smoke": {"cpu": cpu, "gpu": gpu},
        "claim_boundary": {
            "official_source_intake": "ESTABLISHED",
            "cpu_minimum_smoke": "ESTABLISHED",
            "gpu_gqf_path": gpu["status"],
            "paper_exact_reproduction": "NOT_STARTED",
            "surpass_puviani_nmf": "PROHIBITED",
        },
        "bindings": {
            "implementation": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
            "smoke_script": {"path": _relative(SMOKE_SCRIPT), "sha256": _sha256(SMOKE_SCRIPT)},
            "runner_manifest": {"path": _relative(RUNNER_MANIFEST), "sha256": _sha256(RUNNER_MANIFEST)},
            "patch_manifest": {"path": _relative(PATCH_MANIFEST), "sha256": _sha256(PATCH_MANIFEST)},
            "conda_explicit": {"path": _relative(CONDA_EXPLICIT), "sha256": _sha256(CONDA_EXPLICIT)},
            "pip_freeze": {"path": _relative(PIP_FREEZE), "sha256": _sha256(PIP_FREEZE)},
            "history": {"path": _relative(ENV_HISTORY), "sha256": _sha256(ENV_HISTORY)},
            "cpu_smoke": {"path": _relative(CPU_SMOKE_ARTIFACT), "sha256": _sha256(CPU_SMOKE_ARTIFACT)},
            "gpu_stdout": {"path": _relative(GPU_STDOUT), "sha256": _sha256(GPU_STDOUT)},
            "gpu_stderr": {"path": _relative(GPU_STDERR), "sha256": _sha256(GPU_STDERR)},
        },
    }
    report["semantic_mutation_audit"] = {"count": 12, "detected": 12, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "failed": sum(not value for value in report["gates"].values()),
    }
    report["verdict"] = (
        "PASS_GQF_OFFICIAL_INTAKE_" + gpu["status"]
        if all(report["gates"].values())
        else "FAIL_GQF_OFFICIAL_INTAKE"
    )
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    expected_verdict = (
        "PASS_GQF_OFFICIAL_INTAKE_" + report["smoke"]["gpu"]["status"]
        if all(gates.values())
        else "FAIL_GQF_OFFICIAL_INTAKE"
    )
    if report.get("gates") != gates or report.get("verdict") != expected_verdict or not all(gates.values()):
        raise ValueError("T6.8.3 gates/verdict do not recompute")
    if _git("rev-parse", "HEAD").stdout.strip() != report["upstream"]["commit"]:
        raise ValueError("T6.8.3 upstream commit drifted")
    tracked = _tracked_files()
    if _tree_sha256(UPSTREAM_ROOT, tracked) != report["upstream"]["tracked_tree_sha256"]:
        raise ValueError("T6.8.3 upstream tracked tree drifted")
    if _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise ValueError("T6.8.3 upstream tracked checkout is modified")
    for item in report["bindings"].values():
        path = ROOT / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"T6.8.3 bound artifact drifted: {item['path']}")
    patch_files = _patch_files()
    if _patch_series_sha256(patch_files) != report["patch_manifest"]["patch_series_sha256"]:
        raise ValueError("T6.8.3 patch series drifted")
    if not all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]):
        raise ValueError("T6.8.3 mutation audit incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    report = build_report()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_report(json.loads(args.artifact.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "gates": report["gate_summary"],
                "upstream": f"{report['upstream']['url']}@{report['upstream']['commit']}",
                "cpu_smoke": report["smoke"]["cpu"]["status"],
                "gpu_probe": report["smoke"]["gpu"]["status"],
                "patches": len(report["patch_manifest"]["patches"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
