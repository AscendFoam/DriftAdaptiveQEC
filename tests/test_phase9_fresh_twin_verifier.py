from __future__ import annotations

import ast
import copy
import csv
from io import BytesIO
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_fresh_twin_verifier as verifier


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _binding(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": verifier._sha_bytes(payload),
    }


def _component_receipt(
    window: np.ndarray,
    priors: np.ndarray,
    centers: np.ndarray,
    sigma: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    count = len(window)
    squared = np.sum(
        (window[None, :, :] - centers[:, None, :]) ** 2, axis=(1, 2)
    )
    logs = (
        -count * math.log(2.0 * math.pi * sigma * sigma)
        - squared / (2.0 * sigma * sigma)
    )
    weighted = logs + np.log(priors)
    maximum = float(np.max(weighted))
    evidence = maximum + math.log(float(np.sum(np.exp(weighted - maximum))))
    posterior = np.exp(weighted - evidence)
    llr = np.array(
        [logs[0] - logs[1], logs[0] - logs[2], logs[1] - logs[2]]
    ) / count
    residual = float(
        np.sum((window - np.mean(window, axis=0, keepdims=True)) ** 2)
    )
    return evidence, posterior, llr, residual


def _predictive(
    priors: np.ndarray, centers: np.ndarray, sigma: float, count: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = priors @ centers
    centered = centers - mean
    covariance = np.einsum("k,ki,kj->ij", priors, centered, centered)
    covariance += np.eye(2) * sigma * sigma / count
    return mean, covariance


def _base_row(
    *,
    backend: str,
    position: int,
    chunk_id: str,
    raw: np.ndarray,
    heldout: np.ndarray,
    priors: np.ndarray,
    cell_id: str,
    reset: bool = False,
) -> dict[str, object]:
    centers = np.array([[-0.85, 0.0], [0.85, 0.0], [0.0, 1.15]])
    sigma = 0.48
    raw_evidence, posterior, _, residual = _component_receipt(
        raw, priors, centers, sigma
    )
    heldout_evidence, _, heldout_llr, _ = _component_receipt(
        heldout, priors, centers, sigma
    )
    mean, covariance = _predictive(priors, centers, sigma, len(raw))
    row: dict[str, object] = {
        field: "" for field in verifier.EXPECTED_LEDGER_FIELDS
    }
    row.update(
        {
            "row_id": f"{cell_id}|{backend}|p{position:04d}",
            "row_schema": verifier.ROW_SCHEMA,
            "layer": "shared",
            "cell_base": "shared|vacuum_g|RESET" if reset else "shared|vacuum_g|IDLE",
            "cell_id": cell_id,
            "backend": backend,
            "backend_id": f"fixture_backend_{backend.lower()}",
            "cutoff": 8,
            "convergence_role": "same_cutoff_ab",
            "seed": 1_070_000 + position + (1000 if backend == "B" else 0),
            "seed_position": position,
            "trajectory_id": "",
            "round_index": 0,
            "terminal_round": True,
            "action": "RESET" if reset else "IDLE",
            "probe_id": "",
            "scenario": "",
            "initial_state": "vacuum_g",
            "logical_label": "",
            "rng_namespace": f"fixture_{backend}",
            "archive_chunk": chunk_id,
            "archive_row_index": position,
            "density_index": -1,
            "raw_iq_index": position,
            "heldout_iq_index": position,
            "heldout_window_sha256": verifier._sha_bytes(
                np.asarray(heldout, dtype="<f8").tobytes(order="C")
            ),
            "pre_readout_i": 0.0,
            "pre_readout_q": 0.0,
            "pre_measurement_g": priors[0],
            "pre_measurement_e": priors[1],
            "pre_measurement_f": priors[2],
            "pre_reset_g": priors[0],
            "pre_reset_e": priors[1],
            "pre_reset_f": priors[2],
            "integrated_i": float(np.mean(raw[:, 0])),
            "integrated_q": float(np.mean(raw[:, 1])),
            "integrated_i_mean_error": 0.0,
            "integrated_q_mean_error": 0.0,
            "raw_log_evidence": raw_evidence,
            "raw_reference_log_evidence": raw_evidence,
            "raw_within_window_residual": residual,
            "posterior_g": posterior[0],
            "posterior_e": posterior[1],
            "posterior_f": posterior[2],
            "level_g": priors[0],
            "level_e": priors[1],
            "level_f": priors[2],
            "mean_photon": 0.25,
            "reset_requested": reset,
            "reset_hidden_success": False,
            "reset_ack": "not_requested" if not reset else "success",
            "rao_blackwell_reset_success": (
                priors[0] + 0.985 * priors[1] + 0.82 * priors[2]
                if reset
                else ""
            ),
            "leakage_resident": False,
            "leakage_residence_probability": priors[2],
            "leakage_age": 0,
            "predictive_mean_i": mean[0],
            "predictive_mean_q": mean[1],
            "predictive_cov_ii": covariance[0, 0],
            "predictive_cov_iq": covariance[0, 1],
            "predictive_cov_qq": covariance[1, 1],
            "heldout_reference_log_evidence": heldout_evidence,
            "heldout_proper_score_per_sample": heldout_evidence / len(heldout),
            "heldout_llr_ge_per_sample": heldout_llr[0],
            "heldout_llr_gf_per_sample": heldout_llr[1],
            "heldout_llr_ef_per_sample": heldout_llr[2],
            "drift_0": 0.0,
            "drift_1": 0.0,
            "drift_2": 0.0,
            "drift_3": 0.0,
            "drift_4": 0.0,
            "logical_survival": "",
            "density_trace_error": 0.0,
            "density_hermiticity_frobenius": 0.0,
            "density_minimum_eigenvalue": 0.0,
            "density_quantization_frobenius_error": "",
            "density_quantization_certified_frobenius_bound": "",
            "density_quantization_trace_distance_bound": "",
            "posterior_normalization_error": abs(float(np.sum(posterior)) - 1.0),
            "level_normalization_error": 0.0,
            "reference_posterior_l1_error": 0.0,
            "reference_log_evidence_error": 0.0,
            "conservation_pass": True,
            "exception_type": "",
            "exception_message": "",
        }
    )
    return row


def _event(index: int, previous: str, kind: str, **extra: object) -> dict:
    value = {
        "event_kind": kind,
        "task_id": verifier.TASK_ID,
        "run_id": "fixture-run",
        "config_sha256": "1" * 64,
        "seal_sha256": "2" * 64,
        **extra,
        "event_schema": verifier.ATTEMPT_SCHEMA,
        "event_index": index,
        "previous_event_sha256": previous,
    }
    value["event_sha256"] = verifier._sha(value)
    return value


def _write_attempt(path: Path, events: list[dict]) -> bytes:
    payload = b"".join(verifier._canonical(event) + b"\n" for event in events)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _make_fixture(
    root: Path,
    *,
    no_go: bool = False,
    metric: str = "predictive_mean_l2",
    margin: float = 0.01,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs/fresh.json"
    design_path = root / "configs/design.json"
    design_report_path = root / "docs/design.json"
    seal_path = root / "docs/seal.json"
    ledger_path = root / "docs/ledger.csv"
    archive_path = root / "docs/archive.zip"
    manifest_path = root / "docs/manifest.json"
    attempt_path = root / "docs/attempt.jsonl"
    runner_path = root / "src/runner.py"
    auditor_path = root / "src/auditor.py"
    lineage_path = root / "docs/lineage.json"
    for path, payload in (
        (runner_path, b"# fixture runner\n"),
        (auditor_path, b"# fixture auditor\n"),
        (lineage_path, b'{"fixture":"lineage"}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    reset = metric == "rao_blackwell_reset_success"
    source_scope = (
        "ab/c8/reset/vacuum_g"
        if reset
        else "ab/c8/shared/vacuum_g/IDLE"
    )
    gate = {
        "gate_id": f"{source_scope}/{metric}",
        "family": (
            "reset_leakage"
            if reset
            else (
                "likelihood_score_posterior"
                if metric in {
                    "paired_proper_score_per_sample",
                    "pairwise_llr_per_sample",
                    "posterior_mean_l1",
                }
                else "iq_conditional_distribution"
            )
        ),
        "stage": "round",
        "metric": metric,
        "margin": margin,
        "normalized_sd": 1.0,
        "deterministic": False,
        "direction": "upper",
    }
    source_hash = "b" * 64
    design_report = {
        "task_id": verifier.TASK_ID,
        "schema_version": "fixture-design-v1",
        "verdict": "PASS_FRESH_TWIN_DESIGN_POWER",
        "blueprint": {"sha256": source_hash, "gate_count": 1},
    }
    design_report["analysis_sha256"] = verifier._sha(design_report)
    _write_json(design_report_path, design_report)
    design = {
        "task_id": verifier.TASK_ID,
        "statistical_procedure": {
            "global_test": "intersection_union_equivalence",
            "cell_test": "two_one_sided_tests",
            "cluster_unit": (
                "independent seed position; all rows sharing a seed remain together"
            ),
            "raw_log_evidence": (
                "diagnostic only; never a cross-gain primary gate"
            ),
            "mixed_unit_vector_max": False,
            "missing_nonfinite_exception": verifier.INCOMPLETE_VERDICT,
        },
    }
    _write_json(design_path, design)
    blueprint = [gate]
    config = {
        "task_id": verifier.TASK_ID,
        "schema_version": verifier.CONFIG_SCHEMA,
        "formal_result_accessed_before_freeze": False,
        "historical_policy": {
            "historical_no_go_rewritten": False,
            "historical_formal_cell_data_access_allowed": False,
        },
        "design_power": {
            "path": design_report_path.relative_to(root).as_posix(),
            "schema_version": "fixture-design-v1",
            "required_verdict": "PASS_FRESH_TWIN_DESIGN_POWER",
        },
        "preformal_seal": {
            "path": seal_path.relative_to(root).as_posix(),
            "required_status": "PASS_FRESH_TWIN_PREFORMAL_AUDIT_SEALED",
        },
        "common_physics": {
            "iq_samples": 8,
            "iq_sigma": 0.48,
            "iq_centers": [[-0.85, 0.0], [0.85, 0.0], [0.0, 1.15]],
            "reset_success_e": 0.985,
            "reset_success_f": 0.82,
        },
        "artifact_paths": {
            "attempt_ledger": attempt_path.relative_to(root).as_posix()
        },
        "runtime_dependencies": {"paths": []},
        "gate_blueprint": {
            "source_design_blueprint_sha256": source_hash,
            "row_count": 1,
            "canonical_blueprint_sha256": verifier._sha(blueprint),
            "rows": blueprint,
        },
        "verification_contract": {
            "cluster_unit": "seed_position",
            "global_test": "intersection_union_equivalence",
            "cell_test": "two_one_sided_tests",
            "cell_confidence_interval": 0.9,
            "tost_z": verifier.Z_TOST,
            "raw_log_evidence_primary": False,
            "fault_mixed_unit_composite": False,
            "drift_normalization": [0.12, 0.10, 0.18, 0.14, 0.12],
            "aggregate_rescue_forbidden": True,
            "missing_nonfinite_exception": verifier.INCOMPLETE_VERDICT,
            "density_quantization_bound_must_be_added": True,
        },
    }
    _write_json(config_path, config)
    seal = {
        "task_id": verifier.TASK_ID,
        "schema_version": verifier.PREFORMAL_SEAL_SCHEMA,
        "status": "PASS_FRESH_TWIN_PREFORMAL_AUDIT_SEALED",
        "formal_result_accessed": False,
        "historical_formal_cell_data_accessed": False,
        "all_gates_passed": True,
        "all_mutations_detected": True,
        "scientific_verdict": None,
        "live_bindings": {
            "fresh_config": _binding(root, config_path),
            "fresh_runner": _binding(root, runner_path),
            "historical_lineage_receipt": _binding(root, lineage_path),
            "design_power_report": _binding(root, design_report_path),
            "preformal_audit": _binding(root, auditor_path),
        },
    }
    seal["analysis_sha256"] = verifier._sha(seal)
    _write_json(seal_path, seal)

    heldout = np.array(
        [[0.15 + 0.01 * index, -0.20 + 0.005 * index] for index in range(8)]
    )
    rows: list[dict[str, object]] = []
    chunks: list[tuple[str, bytes]] = []
    for backend in ("A", "B"):
        chunk_id = f"chunk-{backend}"
        chunk_rows: list[dict[str, object]] = []
        raw_windows = []
        heldout_windows = []
        for position in range(2):
            raw = np.array(
                [
                    [-0.2 + 0.01 * index + 0.001 * position, 0.1 - 0.004 * index]
                    for index in range(8)
                ]
            )
            priors = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
            if no_go and backend == "B":
                priors = np.array([0.90, 0.05, 0.05], dtype=np.float64)
            cell_id = (
                "ab/c8/shared/vacuum_g/RESET"
                if reset
                else "ab/c8/shared/vacuum_g/IDLE"
            )
            row = _base_row(
                backend=backend,
                position=position,
                chunk_id=chunk_id,
                raw=raw,
                heldout=heldout,
                priors=priors,
                cell_id=cell_id,
                reset=reset,
            )
            chunk_rows.append(row)
            rows.append(row)
            raw_windows.append(raw)
            heldout_windows.append(heldout)
        stream = BytesIO()
        np.savez_compressed(
            stream,
            schema=np.asarray([verifier.ARCHIVE_SCHEMA]),
            chunk_id=np.asarray([chunk_id]),
            cutoff=np.asarray([8], dtype=np.int64),
            row_ids=np.asarray([row["row_id"] for row in chunk_rows]),
            density_row_ids=np.asarray([], dtype="<U1"),
            densities=np.empty((0, 24, 24), dtype=np.complex64),
            raw_iq=np.stack(raw_windows),
            heldout_iq=np.stack(heldout_windows),
        )
        chunks.append((chunk_id, stream.getvalue()))

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=verifier.EXPECTED_LEDGER_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    mapping_stream = BytesIO()
    np.savez_compressed(
        mapping_stream,
        schema=np.asarray([verifier.ARCHIVE_SCHEMA]),
        iq_reference_id=np.asarray(["fixture-reference"]),
    )
    mapping_payload = mapping_stream.getvalue()
    entries = []
    for chunk_id, payload in chunks:
        entries.append(
            {
                "chunk_id": chunk_id,
                "member": f"chunks/{chunk_id}.npz",
                "source": {
                    "path": f"runs/{chunk_id}.npz",
                    "bytes": len(payload),
                    "sha256": verifier._sha_bytes(payload),
                },
            }
        )
    archive_manifest = {
        "task_id": verifier.TASK_ID,
        "schema_version": verifier.ARCHIVE_SCHEMA,
        "entries": entries,
        "mapping_member": "mapping/mapping_arrays.npz",
        "mapping_source": {
            "path": "runs/mapping.npz",
            "bytes": len(mapping_payload),
            "sha256": verifier._sha_bytes(mapping_payload),
        },
    }
    archive_manifest["analysis_sha256"] = verifier._sha(archive_manifest)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            "archive_manifest.json", verifier._canonical(archive_manifest) + b"\n"
        )
        archive.writestr("mapping/mapping_arrays.npz", mapping_payload)
        for (chunk_id, payload) in chunks:
            archive.writestr(f"chunks/{chunk_id}.npz", payload)

    start = _event(0, "0" * 64, "RUN_STARTED")
    committed_a = _event(
        1,
        start["event_sha256"],
        "CHUNK_COMMITTED",
        chunk={"chunk_id": "chunk-A"},
    )
    committed_b = _event(
        2,
        committed_a["event_sha256"],
        "CHUNK_COMMITTED",
        chunk={"chunk_id": "chunk-B"},
    )
    prefix_payload = _write_attempt(
        attempt_path, [start, committed_a, committed_b]
    )
    manifest = {
        "task_id": verifier.TASK_ID,
        "schema_version": verifier.EXECUTION_MANIFEST_SCHEMA,
        "formal": True,
        "status": "FORMAL_RAW_EVIDENCE_COMPLETE",
        "scientific_verdict": None,
        "expected_cells": 2,
        "observed_cells": 2,
        "expected_rows": 4,
        "observed_rows": 4,
        "exception_rows": 0,
        "config": _binding(root, config_path),
        "preformal_seal": _binding(root, seal_path),
        "preformal_seal_analysis_sha256": seal["analysis_sha256"],
        "cell_ledger": _binding(root, ledger_path),
        "raw_archive": _binding(root, archive_path),
        "attempt_ledger_prefix": {
            "path": attempt_path.relative_to(root).as_posix(),
            "bytes": len(prefix_payload),
            "sha256": verifier._sha_bytes(prefix_payload),
            "last_event_index": 2,
            "last_event_sha256": committed_b["event_sha256"],
        },
    }
    manifest["execution_sha256"] = verifier._sha(manifest)
    _write_json(manifest_path, manifest)
    final = _event(
        3,
        committed_b["event_sha256"],
        "FINALIZED",
        execution_manifest=_binding(root, manifest_path),
    )
    _write_attempt(attempt_path, [start, committed_a, committed_b, final])
    return {
        "root": root,
        "config": config_path,
        "design": design_path,
        "ledger": ledger_path,
        "archive": archive_path,
        "manifest": manifest_path,
        "attempt": attempt_path,
    }


def _verify(paths: dict[str, Path]):
    return verifier.verify_bundle(
        paths["root"],
        config_path=paths["config"],
        design_config_path=paths["design"],
        ledger_path=paths["ledger"],
        archive_path=paths["archive"],
        manifest_path=paths["manifest"],
        allow_test_fixture=True,
    )


def _rewrite_attempt_with_run_error(
    paths: dict[str, Path], *, recovered: bool
) -> None:
    start = _event(0, "0" * 64, "RUN_STARTED")
    error = _event(1, start["event_sha256"], "RUN_ERROR")
    events = [start, error]
    previous = error["event_sha256"]
    if recovered:
        resume = _event(2, previous, "RESUME_STARTED")
        events.append(resume)
        previous = resume["event_sha256"]
    commit_a = _event(
        len(events), previous, "CHUNK_COMMITTED", chunk={"chunk_id": "chunk-A"}
    )
    events.append(commit_a)
    commit_b = _event(
        len(events), commit_a["event_sha256"], "CHUNK_COMMITTED",
        chunk={"chunk_id": "chunk-B"},
    )
    events.append(commit_b)
    prefix_payload = _write_attempt(paths["attempt"], events)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["attempt_ledger_prefix"] = {
        "path": paths["attempt"].relative_to(paths["root"]).as_posix(),
        "bytes": len(prefix_payload),
        "sha256": verifier._sha_bytes(prefix_payload),
        "last_event_index": len(events) - 1,
        "last_event_sha256": commit_b["event_sha256"],
    }
    manifest["execution_sha256"] = verifier._sha(
        {key: value for key, value in manifest.items() if key != "execution_sha256"}
    )
    _write_json(paths["manifest"], manifest)
    final = _event(
        len(events),
        commit_b["event_sha256"],
        "FINALIZED",
        execution_manifest=_binding(paths["root"], paths["manifest"]),
    )
    _write_attempt(paths["attempt"], [*events, final])


def test_synthetic_pass_fixture(tmp_path: Path) -> None:
    report, gates, release = _verify(_make_fixture(tmp_path))
    assert report["verdict"] == verifier.PASS_VERDICT
    assert len(gates) == 1 and gates[0]["passed"] is True
    assert release["qualified_claim"] == verifier.QUALIFIED_CLAIM


def test_synthetic_no_go_fixture(tmp_path: Path) -> None:
    report, gates, release = _verify(_make_fixture(tmp_path, no_go=True))
    assert report["verdict"] == verifier.NO_GO_VERDICT
    assert gates[0]["passed"] is False
    assert release["qualified_claim"] is None


def test_synthetic_incomplete_fixture(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    paths["archive"].write_bytes(paths["archive"].read_bytes()[:-7])
    report, gates, release = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT
    assert gates == []
    assert release["qualified_claim"] is None


def test_raw_archive_is_lazy_and_cache_is_bounded(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    archive = verifier.load_archive(paths["archive"])
    assert not isinstance(archive.raw_iq_by_row, dict)
    assert len(archive._cache) == 0
    first = next(iter(archive.raw_iq_by_row))
    assert archive.raw_iq_by_row[first].shape == (8, 2)
    assert 0 < len(archive._cache) <= 4


@pytest.mark.parametrize("verdict", [
    verifier.PASS_VERDICT,
    verifier.NO_GO_VERDICT,
    verifier.INCOMPLETE_VERDICT,
])
def test_all_fifteen_claim_fields_remain_typed_null(verdict: str) -> None:
    release = verifier._release_payload(
        verdict=verdict,
        analysis_sha256="a" * 64 if verdict == verifier.PASS_VERDICT else None,
    )
    assert tuple(release["claim_state"]) == verifier.TYPED_NULL_FIELDS
    assert all(value is None for value in release["claim_state"].values())


def test_only_pass_releases_six_downstream_tasks() -> None:
    for verdict in (verifier.PASS_VERDICT, verifier.NO_GO_VERDICT, verifier.INCOMPLETE_VERDICT):
        release = verifier._release_payload(
            verdict=verdict,
            analysis_sha256="a" * 64 if verdict == verifier.PASS_VERDICT else None,
        )
        assert set(release["downstream_release"]) == set(verifier.DOWNSTREAM_TASKS)
        assert all(
            item["released"] is (verdict == verifier.PASS_VERDICT)
            for item in release["downstream_release"].values()
        )


def test_verifier_has_no_forbidden_imports() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        name.startswith(("physics", "cnn_fpga.benchmark.phase9_fresh_twin_qualification"))
        for name in imports
    )


def test_historical_cell_artifact_path_is_denied(tmp_path: Path) -> None:
    forbidden = "docs/" + next(iter(verifier.PROHIBITED_HISTORICAL_BASENAMES))
    with pytest.raises(verifier.EvidenceIncomplete):
        verifier._safe_relative(tmp_path, forbidden, purpose="mutation")


def test_repository_path_traversal_is_denied(tmp_path: Path) -> None:
    with pytest.raises(verifier.EvidenceIncomplete):
        verifier._safe_relative(tmp_path, "../escape.csv", purpose="mutation")


def test_tost_upper_equality_passes() -> None:
    spec = verifier.GateSpec("g", "f", "round", "m", 0.1)
    result = verifier._result(spec, 0.1, 0.0, 2, "fixture")
    assert result.passed is True and result.bound == 0.1


def test_tost_lower_equality_passes() -> None:
    spec = verifier.GateSpec(
        "g", "f", "round", "principal_singular", 0.95,
        direction="lower", deterministic=True,
    )
    result = verifier._result(spec, 0.95, 0.0, 0, "fixture")
    assert result.passed is True


def test_wrong_bound_direction_fails() -> None:
    with pytest.raises(verifier.EvidenceIncomplete):
        verifier._result(
            verifier.GateSpec("g", "f", "round", "m", 0.1, direction="sideways"),
            0.0, 0.0, 2, "fixture",
        )


def test_density_quantization_certificate_is_added() -> None:
    density = np.eye(2, dtype=np.complex128) / 2
    row = {
        "density_quantization_frobenius_error": 1.0e-7,
        "density_quantization_certified_frobenius_bound": 2.0e-7,
        "density_quantization_trace_distance_bound": 0.5 * math.sqrt(2) * 2.0e-7,
    }
    bound = verifier._density_quantization_bound(row, density)
    estimate, _ = verifier._ensemble_trace_distance(
        [density, density], [density, density], [2 * bound, 2 * bound]
    )
    assert estimate == pytest.approx(2 * bound)


def test_density_quantization_lower_bound_substitution_is_rejected() -> None:
    density = np.eye(2, dtype=np.complex128) / 2
    row = {
        "density_quantization_frobenius_error": 2.0e-7,
        "density_quantization_certified_frobenius_bound": 1.0e-7,
        "density_quantization_trace_distance_bound": 0.5 * math.sqrt(2) * 1.0e-7,
    }
    with pytest.raises(verifier.EvidenceIncomplete):
        verifier._density_quantization_bound(row, density)


def test_density_quantization_dimension_factor_is_checked() -> None:
    density = np.eye(3, dtype=np.complex128) / 3
    row = {
        "density_quantization_frobenius_error": 0.0,
        "density_quantization_certified_frobenius_bound": 1.0e-7,
        "density_quantization_trace_distance_bound": 0.5 * math.sqrt(2) * 1.0e-7,
    }
    with pytest.raises(verifier.EvidenceIncomplete):
        verifier._density_quantization_bound(row, density)


def test_common_heldout_score_uses_backend_specific_priors(tmp_path: Path) -> None:
    report, gates, _ = _verify(
        _make_fixture(
            tmp_path,
            no_go=True,
            metric="paired_proper_score_per_sample",
            margin=1.0e-4,
        )
    )
    assert report["verdict"] == verifier.NO_GO_VERDICT
    assert gates[0]["estimate"] > 0.0


def test_common_heldout_window_mismatch_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(
        tmp_path, metric="paired_proper_score_per_sample", margin=1.0
    )
    # The archive mutation is deliberately not re-signed: integrity failure
    # must classify as INCOMPLETE, never as scientific NO-GO.
    payload = bytearray(paths["archive"].read_bytes())
    payload[-20] ^= 1
    paths["archive"].write_bytes(payload)
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_reset_rao_blackwell_gate_is_recomputed(tmp_path: Path) -> None:
    report, gates, _ = _verify(
        _make_fixture(
            tmp_path,
            metric="rao_blackwell_reset_success",
            margin=0.01,
        )
    )
    assert report["verdict"] == verifier.PASS_VERDICT
    assert gates[0]["estimate"] == pytest.approx(0.0)


def test_duplicate_ledger_row_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    lines = paths["ledger"].read_text(encoding="utf-8").splitlines()
    paths["ledger"].write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_reordered_ledger_header_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    lines = paths["ledger"].read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    header[0], header[1] = header[1], header[0]
    paths["ledger"].write_text(
        ",".join(header) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_nonfinite_ledger_value_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    text = paths["ledger"].read_text(encoding="utf-8")
    text = text.replace(",0.25,", ",nan,", 1)
    paths["ledger"].write_text(text, encoding="utf-8")
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_manifest_scientific_verdict_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["scientific_verdict"] = verifier.PASS_VERDICT
    manifest["execution_sha256"] = verifier._sha(
        {key: value for key, value in manifest.items() if key != "execution_sha256"}
    )
    _write_json(paths["manifest"], manifest)
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_attempt_hash_chain_break_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    payload = paths["attempt"].read_bytes().replace(b"RUN_STARTED", b"RUN_STARTEX", 1)
    paths["attempt"].write_bytes(payload)
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_hash_bound_run_error_followed_by_exact_resume_is_allowed(
    tmp_path: Path,
) -> None:
    paths = _make_fixture(tmp_path)
    _rewrite_attempt_with_run_error(paths, recovered=True)
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.PASS_VERDICT


def test_unrecovered_run_error_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    _rewrite_attempt_with_run_error(paths, recovered=False)
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_blueprint_duplicate_identifier_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["gate_blueprint"]["rows"].append(
        copy.deepcopy(config["gate_blueprint"]["rows"][0])
    )
    config["gate_blueprint"]["row_count"] = 2
    config["gate_blueprint"]["canonical_blueprint_sha256"] = verifier._sha(
        config["gate_blueprint"]["rows"]
    )
    _write_json(paths["config"], config)
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_blueprint_hash_drift_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["gate_blueprint"]["canonical_blueprint_sha256"] = "0" * 64
    _write_json(paths["config"], config)
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_raw_iq_mean_mismatch_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    text = paths["ledger"].read_text(encoding="utf-8")
    header, first, *rest = text.splitlines()
    fields = header.split(",")
    values = first.split(",")
    values[fields.index("integrated_i")] = "9.0"
    paths["ledger"].write_text(
        "\n".join([header, ",".join(values), *rest]) + "\n", encoding="utf-8"
    )
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_exception_row_is_incomplete(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    text = paths["ledger"].read_text(encoding="utf-8")
    header, first, *rest = text.splitlines()
    fields = header.split(",")
    values = first.split(",")
    values[fields.index("exception_type")] = "RuntimeError"
    values[fields.index("exception_message")] = "fixture failure"
    paths["ledger"].write_text(
        "\n".join([header, ",".join(values), *rest]) + "\n", encoding="utf-8"
    )
    report, _, _ = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT


def test_main_rejects_all_overrides() -> None:
    with pytest.raises(SystemExit):
        verifier.main(["--root", "somewhere"])


def test_incomplete_is_not_scientific_no_go(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    paths["manifest"].unlink()
    report, gates, release = _verify(paths)
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT
    assert report["verdict"] != verifier.NO_GO_VERDICT
    assert gates == []
    assert all(
        item["released"] is False
        for item in release["downstream_release"].values()
    )


def test_production_rejects_tiny_fixture_blueprint(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    report, _, _ = verifier.verify_bundle(
        paths["root"],
        config_path=paths["config"],
        design_config_path=paths["design"],
        ledger_path=paths["ledger"],
        archive_path=paths["archive"],
        manifest_path=paths["manifest"],
        allow_test_fixture=False,
    )
    assert report["verdict"] == verifier.INCOMPLETE_VERDICT
