"""Contract tests for the small lazy physics package root."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import physics


EXPECTED_ROOT_EXPORTS = {
    "ApproximateGKPState": "physics.gkp_state",
    "GKPStateFactory": "physics.gkp_state",
    "QuantumNoiseChannel": "physics.noise_channels",
    "PhotonLossChannel": "physics.noise_channels",
    "ThermalNoiseChannel": "physics.noise_channels",
    "SyndromeMeasurement": "physics.syndrome_measurement",
    "RealisticSyndromeMeasurement": "physics.syndrome_measurement",
    "GKPErrorCorrector": "physics.error_correction",
    "LinearDecoder": "physics.error_correction",
    "LogicalErrorTracker": "physics.logical_tracking",
}

PRIVATE_IMPLEMENTATION_MODULES = {
    "physics._control_imperfections.validation",
    "physics._cross_fidelity.reporting",
    "physics._differentiable_sbs.validation",
    "physics._differentiable_sbs.worker",
    "physics._nmf_ranking.execution",
}


def test_root_exports_are_small_lazy_core_aliases() -> None:
    assert physics._EXPORTS == EXPECTED_ROOT_EXPORTS
    assert physics.__all__ == list(EXPECTED_ROOT_EXPORTS)

    for name, module_name in EXPECTED_ROOT_EXPORTS.items():
        assert getattr(physics, name) is getattr(importlib.import_module(module_name), name)


def test_cold_package_import_loads_no_submodule_or_heavy_dependency() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    probe = """
import json
import sys
import physics

heavy = ("numpy", "scipy", "torch", "matplotlib")
print(json.dumps({
    "heavy": {
        package: any(
            name == package or name.startswith(package + ".")
            for name in sys.modules
        )
        for package in heavy
    },
    "physics_submodules": sorted(
        name for name in sys.modules if name.startswith("physics.")
    ),
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip()) == {
        "heavy": {
            "matplotlib": False,
            "numpy": False,
            "scipy": False,
            "torch": False,
        },
        "physics_submodules": [],
    }


def test_all_public_module_paths_remain_importable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    package_dir = Path(physics.__file__).resolve().parent
    public_modules = {
        path.stem
        for path in package_dir.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    }
    failures: dict[str, str] = {}
    qualified_names = {
        *(f"physics.{module_name}" for module_name in public_modules),
        *PRIVATE_IMPLEMENTATION_MODULES,
    }
    for qualified_name in sorted(qualified_names):
        try:
            importlib.import_module(qualified_name)
        except Exception as exc:  # pragma: no cover - diagnostic aggregation
            failures[qualified_name] = f"{type(exc).__name__}: {exc}"
    assert failures == {}
