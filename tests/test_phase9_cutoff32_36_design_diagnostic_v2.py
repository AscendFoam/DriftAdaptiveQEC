from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_cutoff32_36_design_diagnostic as v1
from cnn_fpga.benchmark import phase9_cutoff32_36_design_bootstrap_v2 as b2
from cnn_fpga.benchmark import phase9_cutoff32_36_design_diagnostic_v2 as v2
from cnn_fpga.benchmark import phase9_cutoff32_36_design_extension as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/phase9/t_risk_20260728_01_cutoff32_36_design_extension.json"
)
MANIFEST_PATH = (
    ROOT / "docs/t_risk_20260728_01_cutoff32_36_design_extension_fresh2_manifest.json"
)


def _binding(path: Path, *, semantic: bool = False) -> dict[str, object]:
    payload = path.read_bytes()
    result: dict[str, object] = {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }
    if semantic:
        document = json.loads(payload)
        result["analysis_sha256"] = document["analysis_sha256"]
    return result


def _write_self_hashed_json(path: Path, value: int = 1) -> dict[str, object]:
    payload: dict[str, object] = {"value": value}
    payload["analysis_sha256"] = v2._sha(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def test_v2_accepts_and_checks_preregistered_analysis_binding(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    document = _write_self_hashed_json(artifact)
    payload = artifact.read_bytes()
    binding = {
        "path": artifact.name,
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
        "analysis_sha256": document["analysis_sha256"],
    }

    path, observed = v2._read_bound_bytes(tmp_path, binding)
    assert path == artifact.resolve()
    assert observed == payload

    bad_semantic = deepcopy(binding)
    bad_semantic["analysis_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="semantic binding drift"):
        v2._read_bound_bytes(tmp_path, bad_semantic)

    extra = {**binding, "unexpected": True}
    with pytest.raises(RuntimeError, match="binding schema drift"):
        v2._read_bound_bytes(tmp_path, extra)


def test_v2_bootstrap_is_diagnostic_only_and_release_root_is_self_consistent() -> None:
    assert "phase9_cutoff32_36_design_bootstrap_v2.py" in b2.EXTERNAL_LAUNCHER_SOURCE
    assert (
        sha256(b2.EXTERNAL_LAUNCHER_SOURCE.encode("utf-8")).hexdigest()
        == b2.EXTERNAL_LAUNCHER_SHA256
    )
    assert b2.DIAGNOSTIC_SHA256 == _binding(ROOT / b2.DIAGNOSTIC_PATH)["sha256"]
    b2._verify_release_root(ROOT)
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        b2._commit_launch_meta("pilot")


def test_v2_rejects_analysis_binding_for_non_json(tmp_path: Path) -> None:
    artifact = tmp_path / "not-json.bin"
    artifact.write_bytes(b"not json")
    binding = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": sha256(artifact.read_bytes()).hexdigest(),
        "analysis_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="not JSON"):
        v2._read_bound_bytes(tmp_path, binding)


def test_tail_quantization_bound_respects_observable_lipschitz_factor() -> None:
    certificates = np.asarray([1e-6, 3e-6], dtype=np.float64)
    for metric in (
        "top1_fock_mass",
        "top2_fock_mass",
        "top4_fock_mass",
        "normalized_mean_photon",
    ):
        assert v2._tail_quantization_bound(metric, certificates, 36) == (
            pytest.approx(2e-6)
        )
    assert v2._tail_quantization_bound(
        "commutator_defect", certificates, 36
    ) == pytest.approx(72e-6)
    with pytest.raises(RuntimeError, match="unknown tail metric"):
        v2._tail_quantization_bound("invented", certificates, 36)
    with pytest.raises(RuntimeError, match="certificate drift"):
        v2._tail_quantization_bound("top1_fock_mass", np.asarray([-1e-6]), 36)


def test_live_reference_manifest_reproduces_v1_failure_and_v2_repair() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    binding = config["reference_cutoff_28_evidence"]["manifest"]

    with pytest.raises(RuntimeError, match="binding schema drift"):
        v1._read_bound_bytes(ROOT, binding)

    path, payload = v2._read_bound_bytes(ROOT, binding)
    assert (
        path
        == (ROOT / config["reference_cutoff_28_evidence"]["manifest"]["path"]).resolve()
    )
    assert json.loads(payload)["analysis_sha256"] == binding["analysis_sha256"]

    manifest, selected = v2._selected_reference_receipts(ROOT, config)
    assert manifest["observed_cells"] == 32
    assert len(selected) == 8
    assert {
        (
            receipt["cell"]["scenario"],
            receipt["cell"]["backend"],
            receipt["cell"]["cutoff"],
        )
        for receipt, _binding_value in selected
    } == {
        (scenario, backend, 28)
        for scenario in ("step", "telegraph", "burst", "compound")
        for backend in ("A", "B")
    }


def test_live_production_archive_schema_and_alignment_are_consumed() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    receipt = manifest["chunk_receipts"][0]
    receipt_binding = manifest["receipt_bindings"][0]

    rows, densities = v2._parse_receipt(ROOT, receipt, receipt_binding)
    assert len(rows) == receipt["expected_rows"]
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert len(densities) == sum(row["terminal_round"] for row in rows)
    assert all(
        density.shape == (3 * int(receipt["cell"]["cutoff"]),) * 2
        for density in densities.values()
    )


def test_live_load_evidence_covers_all_30_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v2, "_require_verified_self_import", lambda: None)
    monkeypatch.setattr(v2, "pilot_runner", runner)

    (
        config,
        design_manifest,
        reference_manifest,
        rows,
        densities,
        raw_bindings,
        logical_projection_audit,
    ) = v2._load_evidence(ROOT)

    assert config["task_id"] == "T-RISK-20260728-01"
    assert design_manifest["observed_cells"] == 22
    assert reference_manifest["observed_cells"] == 32
    assert len(rows) == 21_168
    assert len({row["row_id"] for row in rows}) == 21_168
    assert len(densities) == 2_160
    assert len(raw_bindings) == 90
    assert logical_projection_audit["fault_terminal_cross_checks"] == 1_728
    assert logical_projection_audit["shared_terminal_derivations"] == 432
    assert logical_projection_audit["maximum_fault_absolute_delta"] <= (
        logical_projection_audit["maximum_fault_allowed_delta"]
    )
    gates = v2.evaluate(config, rows, densities)
    assert len(gates) == 1_454
    assert len({gate["gate_id"] for gate in gates}) == 1_454
    density_derived = [
        gate
        for gate in gates
        if gate["family"]
        in {
            "fault_density",
            "fault_absolute_tail",
            "shared_density",
            "shared_absolute_tail",
        }
        or (gate["family"] == "shared_scalar" and gate["metric"] == "logical_survival")
    ]
    assert density_derived
    assert all(float(gate["quantization_bound"]) > 0.0 for gate in density_derived)
