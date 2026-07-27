from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from cnn_fpga.benchmark import phase9_high_cutoff_design_pilot as subject
from cnn_fpga.benchmark import phase9_fresh_twin_qualification as runner


ROOT = Path(__file__).resolve().parents[1]
PREREGISTERED_BOOTSTRAP_SHA256 = (
    "a0a873b6328b05a5070da39b710f7c7e780fb12d4c1aad29ed3323fbdf3405ee"
)
PREREGISTERED_EXTERNAL_LAUNCHER_SHA256 = (
    "e732eb9fec98ed5955f1864feeffde33a364fc79b011ac4b83ae48c872e97be6"
)
PREREGISTERED_EXTERNAL_LAUNCHER_SOURCE = (
    "import hashlib,json,os,pathlib,sys\n"
    "root=pathlib.Path(sys.argv[1]).resolve()\n"
    "expected=sys.argv[2]\n"
    "launcher_sha=sys.argv[3]\n"
    "mode=sys.argv[4]\n"
    "actual_source=sys.orig_argv[sys.orig_argv.index('-c')+1]\n"
    "assert hashlib.sha256(actual_source.encode('utf-8')).hexdigest()==launcher_sha\n"
    "path=root/'cnn_fpga/benchmark/phase9_high_cutoff_design_bootstrap.py'\n"
    "payload=path.read_bytes()\n"
    "assert hashlib.sha256(payload).hexdigest()==expected\n"
    "sys.path.insert(0,str(root))\n"
    "sys.path.append(str(pathlib.Path(sys.base_prefix)/'Lib'/'site-packages'))\n"
    "dll=pathlib.Path(sys.base_prefix)/'Library'/'bin'\n"
    "dll_handle=os.add_dll_directory(str(dll)) if dll.is_dir() else None\n"
    "binding={'path':path.relative_to(root).as_posix(),"
    "'bytes':len(payload),'sha256':expected}\n"
    "namespace={'__name__':'__main__','__file__':str(path),"
    "'__package__':'cnn_fpga.benchmark',"
    "'__verified_source_sha256__':expected,"
    "'__verified_external_launcher_sha256__':launcher_sha,"
    "'__verified_external_launcher_source__':actual_source,"
    "'__verified_external_launcher_flags__':('-I','-S'),"
    "'__verified_bootstrap_source_binding__':binding}\n"
    "sys.argv=[str(path),mode]\n"
    "exec(compile(payload,str(path),'exec',dont_inherit=True),namespace)\n"
)


def _external_bootstrap_probe(root: Path = ROOT) -> dict[str, object]:
    bootstrap_path = root / "cnn_fpga/benchmark/phase9_high_cutoff_design_bootstrap.py"
    expected_sha256 = PREREGISTERED_BOOTSTRAP_SHA256
    launcher_source = PREREGISTERED_EXTERNAL_LAUNCHER_SOURCE
    launcher_sha256 = PREREGISTERED_EXTERNAL_LAUNCHER_SHA256
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            launcher_source,
            str(root),
            expected_sha256,
            launcher_sha256,
            "probe",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module", autouse=True)
def _activate_verified_execution_dependencies():
    bootstrap_path = ROOT / "cnn_fpga/benchmark/phase9_high_cutoff_design_bootstrap.py"
    launch_path = (
        ROOT / "runs/t_risk_20260727_01_test_audit/" "unit_test_pilot_launch_meta.json"
    )
    launch_payload = {"fixture": "verified-launch-meta-content"}
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    launch_path.write_text(
        json.dumps(launch_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subject.__verified_bootstrap_source_binding__ = subject._binding(
        bootstrap_path,
        ROOT,
    )
    subject.__verified_launch_meta_binding__ = subject._binding(launch_path, ROOT)
    subject.__verified_launch_meta_payload__ = launch_payload
    try:
        pilot, _base = subject.load_pilot_config(ROOT)
        snapshot = subject._build_input_snapshot(ROOT, pilot)
        subject._activate_verified_execution_modules(ROOT, snapshot)
        yield
    finally:
        launch_path.unlink(missing_ok=True)


def _released_fixture(value: dict) -> subject.ReleasedPilotConfig:
    binding = {"path": "fixture", "bytes": 1, "sha256": "bound"}
    return subject.ReleasedPilotConfig(
        value,
        release_lineage={
            "released_child": binding,
            "pending_parent": binding,
            "release_receipt": binding,
            "release_receipt_analysis_sha256": "receipt-hash",
            "hardened_confirmation_report": binding,
            "hardened_confirmation_source_data": binding,
            "hardened_confirmation_analysis_sha256": (
                subject.HARDENED_CONFIRMATION_ANALYSIS_SHA256
            ),
            "authorization_state": subject.NARROW_AUTHORIZATION_STATE,
            "narrow_scope": dict(subject.NARROW_SCOPE),
        },
    )


def _spawn_release_roundtrip(
    config: subject.ReleasedPilotConfig,
) -> tuple[str, dict, dict]:
    return type(config).__name__, dict(config), config.release_lineage


def test_live_config_has_bound_sources_and_disjoint_pilot_matrix() -> None:
    pilot, base = subject.load_pilot_config(ROOT)
    execution = subject.materialize_execution_config(pilot, base)
    cells = subject.build_pilot_cells(pilot, execution)

    assert len(cells) == 32
    assert {cell.cutoff for cell in cells} == {16, 20, 24, 28}
    assert {cell.backend for cell in cells} == {"A", "B"}
    assert {cell.scenario for cell in cells} == {
        "step",
        "telegraph",
        "burst",
        "compound",
    }
    assert all(cell.sample_count == 72 for cell in cells)
    assert all(cell.expected_rows == 864 for cell in cells)
    assert sum(cell.expected_rows for cell in cells) == 27648

    first_a = next(cell for cell in cells if cell.backend == "A")
    first_b = next(cell for cell in cells if cell.backend == "B")
    assert runner._seed_for(execution, first_a, 0) == 1430000
    assert runner._seed_for(execution, first_b, 0) == 1431000
    assert execution["formal_splits"]["heldout_common"]["start"] == 1432000
    assert pilot["claim_boundary"]["twin_qualification"] is None
    assert pilot["optional_cutoff_32"]["enabled"] is False
    assert pilot["diagnostic_contract"]["multiplier_replicates"] == 199
    assert pilot["diagnostic_contract"]["multiplier_seed_namespace"] == 1420000
    assert pilot["diagnostic_contract"]["formal_rescue_forbidden"] is True
    for partition in pilot["stage_partition"].values():
        rounds = [
            round_index for indices in partition.values() for round_index in indices
        ]
        assert sorted(rounds) == list(range(12))
        assert len(rounds) == len(set(rounds)) == 12
    snapshot = subject._build_input_snapshot(ROOT, pilot)
    assert len(snapshot) == 16
    assert "verified_bootstrap_source" in snapshot
    assert "verified_external_launch_meta" in snapshot
    assert {
        key.removeprefix("source/") for key in snapshot if key.startswith("source/")
    } == set(pilot["source_bindings"])
    subject._assert_input_snapshot(ROOT, snapshot)


def test_live_high_cutoff_preflight_executes_production_paths_and_convergence() -> None:
    pilot, base = subject.load_pilot_config(ROOT)
    execution = subject.materialize_execution_config(pilot, base)

    report = subject._run_high_cutoff_preflight(pilot, execution)

    assert report["status"] == "PASS"
    assert report["evaluated_cutoffs"] == [16, 20, 24, 28, 32]
    assert report["production_segment_steps"] == 8
    assert report["production_iq_samples"] == 8
    assert len(report["checks"]) == 10
    assert len(report["integration_convergence"]) == 4
    assert all(
        check["high_energy_actions_executed"] == (3 if check["cutoff"] >= 28 else 0)
        for check in report["checks"]
    )
    assert all(
        check["trace_distance_16_to_32"]
        < check["trace_distance_8_to_16"]
        and check["all_checks_passed"] is True
        for check in report["integration_convergence"]
    )
    assert report["qualified_claim"] is None
    assert all(
        value is None
        for key, value in report["claim_state"].items()
        if key != "design_pilot_only"
    )


def test_high_cutoff_preflight_mutations_fail_closed(monkeypatch) -> None:
    pilot, base = subject.load_pilot_config(ROOT)
    execution = subject.materialize_execution_config(pilot, base)
    report = subject._run_high_cutoff_preflight(pilot, execution)

    for mutate, expected in (
        (
            lambda value: value.update(production_segment_steps=1),
            "contract drift",
        ),
        (
            lambda value: value["checks"].pop(),
            "contract drift",
        ),
        (
            lambda value: value["integration_convergence"][0].update(
                trace_distance_16_to_32=0.01
            ),
            "convergence failed",
        ),
    ):
        corrupted = copy.deepcopy(report)
        mutate(corrupted)
        corrupted.pop("analysis_sha256")
        corrupted["analysis_sha256"] = subject._sha(corrupted)
        with pytest.raises(RuntimeError, match=expected):
            subject._validate_high_cutoff_preflight(pilot, execution, corrupted)

    broken_execution = copy.deepcopy(execution)
    broken_execution["common_physics"]["segment_steps"] = 1
    with pytest.raises(RuntimeError, match="contract drift"):
        subject._validate_high_cutoff_preflight(pilot, broken_execution, report)


def test_materialization_does_not_mutate_bound_base_config() -> None:
    pilot, base = subject.load_pilot_config(ROOT)
    before = json.dumps(base, sort_keys=True)
    execution = subject.materialize_execution_config(pilot, base)
    assert json.dumps(base, sort_keys=True) == before
    assert execution is not base
    assert execution["formal_matrix"]["trajectory_sample_count"] == 72


def test_released_child_authorizes_only_narrow_exploratory_localization() -> None:
    config, _base = subject.load_pilot_config(ROOT, require_hardened=True)
    lineage = subject._release_lineage(config)
    assert lineage["authorization_state"] == subject.NARROW_AUTHORIZATION_STATE
    assert lineage["narrow_scope"] == subject.NARROW_SCOPE
    assert lineage["narrow_scope"]["downstream_release"] is False
    assert lineage["narrow_scope"]["physical_coverage_guarantee"] is None


def test_pending_parent_is_byte_immutable_and_only_pins_are_materialized() -> None:
    pending_path = ROOT / subject.PENDING_CONFIG_PATH
    before = pending_path.read_bytes()
    assert len(before) == subject.PENDING_CONFIG_BYTES
    assert sha256(before).hexdigest() == subject.PENDING_CONFIG_SHA256
    pending = json.loads(before)

    materialized, _base = subject.load_pilot_config(ROOT)

    assert pending_path.read_bytes() == before
    assert subject._leaf_differences(pending, materialized) == {
        ("hardened_confirmation_source", "report", "bytes"),
        ("hardened_confirmation_source", "report", "sha256"),
        ("hardened_confirmation_source", "source_data", "bytes"),
        ("hardened_confirmation_source", "source_data", "sha256"),
        ("hardened_confirmation_source", "required_analysis_sha256"),
    }


def test_release_artifact_external_anchors_match_live_bytes() -> None:
    child_path = ROOT / subject.CONFIG_PATH
    receipt_path = ROOT / subject.RELEASE_RECEIPT_PATH
    child = json.loads(child_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert child_path.stat().st_size == subject.RELEASED_CHILD_BYTES
    assert sha256(child_path.read_bytes()).hexdigest() == subject.RELEASED_CHILD_SHA256
    assert child["analysis_sha256"] == subject.RELEASED_CHILD_ANALYSIS_SHA256
    assert receipt_path.stat().st_size == subject.RELEASE_RECEIPT_BYTES
    assert (
        sha256(receipt_path.read_bytes()).hexdigest() == subject.RELEASE_RECEIPT_SHA256
    )
    assert receipt["analysis_sha256"] == subject.RELEASE_RECEIPT_ANALYSIS_SHA256


def test_released_child_rejects_non_whitelisted_patch(monkeypatch) -> None:
    child = json.loads((ROOT / subject.CONFIG_PATH).read_text(encoding="utf-8"))
    child["pin_patch"]["trajectory_count"] = 999
    child.pop("analysis_sha256")
    child["analysis_sha256"] = subject._sha(child)
    monkeypatch.setattr(
        subject, "RELEASED_CHILD_ANALYSIS_SHA256", child["analysis_sha256"]
    )

    with pytest.raises(RuntimeError, match="pin patch whitelist drift"):
        subject._materialize_released_parent(ROOT, child)


def test_release_receipt_rejects_rehashed_claim_tamper(monkeypatch) -> None:
    child = json.loads((ROOT / subject.CONFIG_PATH).read_text(encoding="utf-8"))
    receipt = json.loads(
        (ROOT / subject.RELEASE_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    receipt["qualified_claim"] = "forbidden-upgrade"
    receipt.pop("analysis_sha256")
    receipt["analysis_sha256"] = subject._sha(receipt)
    monkeypatch.setattr(
        subject, "RELEASE_RECEIPT_ANALYSIS_SHA256", receipt["analysis_sha256"]
    )

    with pytest.raises(RuntimeError, match="receipt semantic drift"):
        subject._validate_release_receipt(
            receipt,
            pending_binding=child["pending_parent"],
            report_binding=child["hardened_confirmation"]["report"],
            source_binding=child["hardened_confirmation"]["source_data"],
        )


def test_receipt_analysis_and_byte_bindings_fail_closed(tmp_path, monkeypatch) -> None:
    pilot = {
        "artifact_paths": {"receipt_directory": "receipts"},
        "purpose": "fixture",
    }
    cell = runner.CellSpec(
        chunk_id="chunk",
        layer="fault",
        cell_base="fault|step",
        cutoff=16,
        backend="A",
        sample_count=6,
        convergence_role="pilot",
        scenario="step",
        horizon=12,
    )
    csv_path = tmp_path / "chunk.csv"
    npz_path = tmp_path / "chunk.npz"
    csv_path.write_bytes(b"csv")
    npz_path.write_bytes(b"npz")
    monkeypatch.setattr(
        subject.runner,
        "_validate_chunk_files",
        lambda *_args: None,
    )
    run_identity = {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "analysis_sha256": "identity-hash",
        "input_snapshot_analysis_sha256": "snapshot-hash",
        "pilot_source_sha256": subject._pilot_source_sha256(),
    }
    execution_analysis_sha256 = "execution-hash"
    receipt = {
        "task_id": subject.TASK_ID,
        "schema_version": subject.RECEIPT_SCHEMA,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "config_analysis_sha256": subject._sha(pilot),
        "execution_analysis_sha256": execution_analysis_sha256,
        "input_snapshot_analysis_sha256": "snapshot-hash",
        "pilot_source_sha256": sha256(Path(subject.__file__).read_bytes()).hexdigest(),
        "cell": asdict(cell),
        "chunk_id": cell.chunk_id,
        "cell_base": cell.cell_base,
        "layer": cell.layer,
        "backend": cell.backend,
        "cutoff": cell.cutoff,
        "expected_rows": cell.expected_rows,
        "observed_rows": cell.expected_rows,
        "exception_rows": 0,
        "csv": subject._binding(csv_path, tmp_path),
        "npz": subject._binding(npz_path, tmp_path),
    }
    receipt["analysis_sha256"] = subject._sha(receipt)
    subject._validate_receipt(
        tmp_path,
        pilot,
        cell,
        receipt,
        run_identity=run_identity,
        execution_analysis_sha256=execution_analysis_sha256,
    )

    csv_path.write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="csv binding drift"):
        subject._validate_receipt(
            tmp_path,
            pilot,
            cell,
            receipt,
            run_identity=run_identity,
            execution_analysis_sha256=execution_analysis_sha256,
        )


def test_owner_lock_rejects_second_supervisor_and_cleans_up(tmp_path) -> None:
    pilot = {"artifact_paths": {"owner_lock": "run/supervisor.owner.lock"}}
    lock_path = tmp_path / "run" / "supervisor.owner.lock"
    with subject._exclusive_owner_lock(tmp_path, pilot):
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="owner lock already exists"):
            with subject._exclusive_owner_lock(tmp_path, pilot):
                pass
    assert not lock_path.exists()


def test_owner_lock_disappearance_fails_closed(tmp_path) -> None:
    pilot = {"artifact_paths": {"owner_lock": "run/supervisor.owner.lock"}}
    lock_path = tmp_path / "run" / "supervisor.owner.lock"

    with pytest.raises(RuntimeError, match="owner lock disappeared"):
        with subject._exclusive_owner_lock(tmp_path, pilot) as owner:
            subject._assert_owner_lock(tmp_path, pilot, owner)
            lock_path.unlink()


def test_input_snapshot_detects_late_bound_source_tamper(tmp_path, monkeypatch) -> None:
    pilot_source = tmp_path / "pilot.py"
    pilot_source.write_text("PILOT = 1\n", encoding="utf-8")
    fixture = tmp_path / "bound.py"
    fixture.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = {
        "pilot_source": subject._binding(pilot_source, tmp_path),
        "source/fixture": subject._binding(fixture, tmp_path),
    }
    monkeypatch.setattr(
        subject,
        "_pilot_source_sha256",
        lambda: snapshot["pilot_source"]["sha256"],
    )
    monkeypatch.setattr(subject, "_VERIFIED_EXECUTION_BINDINGS", {})
    monkeypatch.setattr(subject, "_VERIFIED_EXECUTION_MODULES", {})
    subject._assert_input_snapshot(tmp_path, snapshot)

    fixture.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact byte binding drift"):
        subject._assert_input_snapshot(tmp_path, snapshot)


def test_verified_loader_rejects_self_restoring_transient_import_swap(
    tmp_path, monkeypatch
) -> None:
    original_modules = {
        module_name: sys.modules.get(module_name)
        for module_name in subject._EXECUTION_MODULE_NAMES.values()
    }
    pilot_source = tmp_path / "pilot.py"
    pilot_source.write_text("PILOT = 1\n", encoding="utf-8")
    fixture = tmp_path / "bound.py"
    trusted_source = "EVIL = False\n"
    fixture.write_text(trusted_source, encoding="utf-8")
    snapshot = {
        "pilot_source": subject._binding(pilot_source, tmp_path),
        "source/fresh_runner": subject._binding(fixture, tmp_path),
    }
    monkeypatch.setattr(
        subject,
        "_pilot_source_sha256",
        lambda: snapshot["pilot_source"]["sha256"],
    )
    monkeypatch.setattr(
        subject,
        "_EXECUTION_MODULE_NAMES",
        {"fresh_runner": "phase9_verified_loader_attack_fixture"},
    )
    monkeypatch.setattr(subject, "_VERIFIED_EXECUTION_BINDINGS", {})
    monkeypatch.setattr(subject, "_VERIFIED_EXECUTION_MODULES", {})
    monkeypatch.setattr(subject, "runner", None)
    attacker_source = (
        "from pathlib import Path\n"
        "EVIL = True\n"
        f"Path(__file__).write_text({trusted_source!r}, encoding='utf-8')\n"
    )
    fixture.write_text(attacker_source, encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact byte binding drift"):
        subject._activate_verified_execution_modules(tmp_path, snapshot)
    assert "phase9_verified_loader_attack_fixture" not in sys.modules
    assert fixture.read_text(encoding="utf-8") == attacker_source

    fixture.write_text(trusted_source, encoding="utf-8")
    subject._activate_verified_execution_modules(tmp_path, snapshot)
    assert subject.runner.EVIL is False
    assert (
        subject.runner.__verified_source_sha256__
        == snapshot["source/fresh_runner"]["sha256"]
    )
    sys.modules.pop("phase9_verified_loader_attack_fixture", None)
    for module_name, module in original_modules.items():
        if module is None:
            continue
        sys.modules[module_name] = module
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attribute, module)


def test_released_pilot_config_survives_spawn_pickle_roundtrip() -> None:
    config = _released_fixture({"purpose": "spawn"})

    restored = pickle.loads(pickle.dumps(config))

    assert isinstance(restored, subject.ReleasedPilotConfig)
    assert restored == config
    assert restored.release_lineage == config.release_lineage


def test_released_pilot_config_survives_real_spawn_worker() -> None:
    config = _released_fixture({"purpose": "spawn-worker"})

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        type_name, value, lineage = executor.submit(
            _spawn_release_roundtrip, config
        ).result(timeout=30)

    assert type_name == "ReleasedPilotConfig"
    assert value == config
    assert lineage == config.release_lineage


def test_run_identity_rejects_synchronized_tamper(tmp_path, monkeypatch) -> None:
    pilot = {
        "artifact_paths": {"run_identity": "run/run_identity.json"},
        "purpose": "fixture",
    }
    pilot = _released_fixture(pilot)
    execution_hash = "execution-hash"
    snapshot = {"pilot_source": {"path": "fixture", "bytes": 1, "sha256": "x"}}
    monkeypatch.setattr(subject, "_assert_input_snapshot", lambda *_a, **_k: None)
    identity = subject._load_or_create_run_identity(
        tmp_path,
        pilot,
        execution_hash,
        snapshot,
    )
    assert identity["run_id"]
    path = tmp_path / "run" / "run_identity.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["config_analysis_sha256"] = "attacker-rehash"
    unsigned = dict(tampered)
    unsigned.pop("analysis_sha256")
    tampered["analysis_sha256"] = subject._sha(unsigned)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="run identity binding drift"):
        subject._load_or_create_run_identity(
            tmp_path,
            pilot,
            execution_hash,
            snapshot,
        )


def test_manifest_rejects_rehashed_claim_tamper(tmp_path, monkeypatch) -> None:
    pilot = {
        "base_config": {"path": "base.json"},
        "source_bindings": {},
        "artifact_paths": {"run_identity": "run_identity.json"},
        "hardened_confirmation_source": {
            "report": {"path": "hardened.json"},
            "source_data": {"path": "hardened.csv"},
        },
        "claim_boundary": dict(subject.CLAIM_BOUNDARY),
    }
    pilot = _released_fixture(pilot)
    execution = {"fixture": True}
    cell = runner.CellSpec(
        chunk_id="chunk",
        layer="fault",
        cell_base="fault|step",
        cutoff=16,
        backend="A",
        sample_count=1,
        convergence_role="pilot",
        scenario="step",
        horizon=1,
    )
    input_snapshot = {
        "pilot_source": {"fixture": True},
        "verified_bootstrap_source": {
            "path": "bootstrap.py",
            "bytes": 1,
            "sha256": "bound",
        },
        "verified_external_launch_meta": {
            "path": "launch-meta.json",
            "bytes": 1,
            "sha256": "bound",
        },
    }
    run_identity = {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "analysis_sha256": "identity-hash",
        "input_snapshot": input_snapshot,
        "input_snapshot_analysis_sha256": subject._sha(input_snapshot),
        "pilot_source_sha256": subject._pilot_source_sha256(),
    }
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda path, root: {
            "path": Path(path).name,
            "bytes": 1,
            "sha256": "bound",
        },
    )
    monkeypatch.setattr(subject, "_validate_receipt_file", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_chunk_health", lambda *_a, **_k: (0, 0))
    monkeypatch.setattr(subject, "_assert_input_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        subject,
        "_validate_high_cutoff_preflight",
        lambda *_a, **_k: None,
    )
    bindings = {
        "config": subject._binding(tmp_path / subject.CONFIG_PATH, tmp_path),
        "pending_config": dict(subject._release_lineage(pilot)["pending_parent"]),
        "release_receipt": dict(subject._release_lineage(pilot)["release_receipt"]),
        "base_config": subject._binding(tmp_path / "base.json", tmp_path),
        "pilot_source": subject._binding(Path(subject.__file__), tmp_path),
        "run_identity": subject._binding(tmp_path / "run_identity.json", tmp_path),
        "hardened_confirmation_report": subject._binding(
            tmp_path / "hardened.json", tmp_path
        ),
        "hardened_confirmation_source_data": subject._binding(
            tmp_path / "hardened.csv", tmp_path
        ),
        "verified_bootstrap_source": dict(input_snapshot["verified_bootstrap_source"]),
        "verified_external_launch_meta": dict(
            input_snapshot["verified_external_launch_meta"]
        ),
    }
    manifest = {
        "task_id": subject.TASK_ID,
        "schema_version": subject.MANIFEST_SCHEMA,
        "status": subject.STATUS,
        "scientific_verdict": None,
        "qualified_claim": None,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "config_analysis_sha256": subject._sha(pilot),
        "execution_analysis_sha256": subject._sha(execution),
        "input_snapshot_analysis_sha256": run_identity[
            "input_snapshot_analysis_sha256"
        ],
        "pilot_source_sha256": sha256(Path(subject.__file__).read_bytes()).hexdigest(),
        "release_lineage": subject._release_lineage(pilot),
        "observed_cells": 1,
        "observed_rows": cell.expected_rows,
        "exception_rows": 0,
        "conservation_failure_rows": 0,
        "chunk_receipts": [{"cell": {"chunk_id": cell.chunk_id}}],
        "receipt_bindings": [{"path": "receipt.json"}],
        "capability_preflight": {"fixture": True},
        "claim_state": dict(subject.CLAIM_BOUNDARY),
        "bindings": bindings,
        "runtime": {},
    }
    manifest["analysis_sha256"] = subject._sha(manifest)
    subject._verify_manifest(
        tmp_path,
        pilot,
        execution,
        [cell],
        run_identity,
        manifest,
    )

    tampered = dict(manifest)
    tampered["qualified_claim"] = "attacker-upgrade"
    tampered.pop("analysis_sha256")
    tampered["analysis_sha256"] = subject._sha(tampered)
    with pytest.raises(RuntimeError, match="semantic drift"):
        subject._verify_manifest(
            tmp_path,
            pilot,
            execution,
            [cell],
            run_identity,
            tampered,
        )

    missing_trust_binding = json.loads(json.dumps(manifest))
    missing_trust_binding["bindings"].pop("verified_external_launch_meta")
    missing_trust_binding.pop("analysis_sha256")
    missing_trust_binding["analysis_sha256"] = subject._sha(missing_trust_binding)
    with pytest.raises(RuntimeError, match="live binding drift"):
        subject._verify_manifest(
            tmp_path,
            pilot,
            execution,
            [cell],
            run_identity,
            missing_trust_binding,
        )


def test_help_exits_before_runner(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(subject, "run_pilot", lambda *_args: calls.append(True))
    with pytest.raises(SystemExit) as raised:
        subject.main(["--help"])
    assert raised.value.code == 0
    assert calls == []


def _install_run_pilot_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    duplicate_receipt_set: bool = False,
) -> tuple[list[runner.CellSpec], Path, Path]:
    pilot = {
        "max_workers": 2,
        "base_config": {"path": "base.json"},
        "source_bindings": {},
        "artifact_paths": {
            "execution_manifest": "run/manifest.json",
            "heartbeat": "run/heartbeat.json",
            "owner_lock": subject.OWNER_LOCK_PATH,
            "receipt_directory": "run/receipts",
            "run_identity": "run/run_identity.json",
        },
        "hardened_confirmation_source": {
            "report": {"path": "hardened.json"},
            "source_data": {"path": "hardened.csv"},
        },
        "claim_boundary": dict(subject.CLAIM_BOUNDARY),
    }
    pilot = _released_fixture(pilot)
    execution = {"fixture": True}
    cells = [
        runner.CellSpec(
            chunk_id=f"chunk-{backend}",
            layer="fault",
            cell_base=f"fault|step|{backend}",
            cutoff=16,
            backend=backend,
            sample_count=1,
            convergence_role="pilot",
            scenario="step",
            horizon=1,
        )
        for backend in ("A", "B")
    ]
    input_snapshot = {
        "pilot_source": {"fixture": True},
        "verified_bootstrap_source": {
            "path": "bootstrap.py",
            "bytes": 1,
            "sha256": "bound",
        },
        "verified_external_launch_meta": {
            "path": "launch-meta.json",
            "bytes": 1,
            "sha256": "bound",
        },
    }
    run_identity = {
        "run_id": "00000000-0000-0000-0000-000000000001",
        "analysis_sha256": "identity-hash",
        "input_snapshot": input_snapshot,
        "input_snapshot_analysis_sha256": subject._sha(input_snapshot),
        "pilot_source_sha256": subject._pilot_source_sha256(),
    }
    manifest_path = tmp_path / pilot["artifact_paths"]["execution_manifest"]

    def receipt_for(cell: runner.CellSpec) -> dict:
        selected = cells[0] if duplicate_receipt_set else cell
        return {
            "cell": asdict(selected),
            "csv": {"path": f"run/{selected.chunk_id}.csv"},
            "npz": {"path": f"run/{selected.chunk_id}.npz"},
        }

    class DoneFuture:
        def __init__(self, receipt: dict) -> None:
            self._receipt = receipt

        def result(self) -> dict:
            return self._receipt

    class ImmediatePool:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def submit(self, _function, *args):
            cell = runner.CellSpec(**args[3])
            return DoneFuture(receipt_for(cell))

    monkeypatch.setattr(
        subject, "load_pilot_config", lambda *_a, **_k: (pilot, {"base": True})
    )
    monkeypatch.setattr(
        subject, "materialize_execution_config", lambda *_a, **_k: execution
    )
    monkeypatch.setattr(subject, "build_pilot_cells", lambda *_a, **_k: cells)
    monkeypatch.setattr(
        subject, "_exclusive_owner_lock", lambda *_a, **_k: nullcontext()
    )
    monkeypatch.setattr(subject, "_assert_owner_lock", lambda *_a, **_k: None)
    monkeypatch.setattr(
        subject,
        "_build_input_snapshot",
        lambda *_a, **_k: run_identity["input_snapshot"],
    )
    monkeypatch.setattr(subject, "_assert_input_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        subject,
        "_run_high_cutoff_preflight",
        lambda *_a, **_k: {"fixture": True},
    )
    monkeypatch.setattr(
        subject,
        "_validate_high_cutoff_preflight",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(subject, "_require_verified_self_import", lambda: None)
    monkeypatch.setattr(
        subject,
        "_activate_verified_execution_modules",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        subject, "_load_or_create_run_identity", lambda *_a, **_k: run_identity
    )
    monkeypatch.setattr(subject, "ThreadPoolExecutor", ImmediatePool)
    monkeypatch.setattr(subject, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        subject,
        "_binding",
        lambda path, _root: {
            "path": Path(path).name,
            "bytes": 1,
            "sha256": "bound",
        },
    )
    monkeypatch.setattr(
        subject,
        "_read_bound_json",
        lambda *_a, **_k: (
            manifest_path,
            json.loads(manifest_path.read_text(encoding="utf-8")),
        ),
    )
    monkeypatch.setattr(subject, "_validate_receipt_file", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_chunk_health", lambda *_a, **_k: (0, 0))
    return (
        cells,
        manifest_path,
        tmp_path / pilot["artifact_paths"]["heartbeat"],
    )


def _assert_failed_heartbeat(
    heartbeat_path: Path,
    *,
    error_type: str,
) -> None:
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["active"] is False
    assert heartbeat["state"] == "FAILED"
    assert heartbeat["error_type"] == error_type
    assert heartbeat["analysis_sha256"] == subject._sha(
        {key: value for key, value in heartbeat.items() if key != "analysis_sha256"}
    )


def test_preflight_failure_occurs_before_any_scientific_worker(
    tmp_path,
    monkeypatch,
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path,
        monkeypatch,
    )
    worker_pool_started = False

    class ForbiddenPool:
        def __init__(self, **_kwargs) -> None:
            nonlocal worker_pool_started
            worker_pool_started = True
            raise AssertionError("scientific worker pool must remain unopened")

    monkeypatch.setattr(subject, "ThreadPoolExecutor", ForbiddenPool)
    monkeypatch.setattr(
        subject,
        "_run_high_cutoff_preflight",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("capability preflight rejected")
        ),
    )

    with pytest.raises(RuntimeError, match="capability preflight rejected"):
        subject.run_pilot(tmp_path)

    assert worker_pool_started is False
    assert not manifest_path.exists()
    _assert_failed_heartbeat(heartbeat_path, error_type="RuntimeError")


def test_receipt_set_drift_finalization_fails_closed(tmp_path, monkeypatch) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path,
        monkeypatch,
        duplicate_receipt_set=True,
    )

    with pytest.raises(RuntimeError, match="receipt cell set drift"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="RuntimeError")
    assert not manifest_path.exists()


def test_chunk_health_exception_finalization_fails_closed(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        subject,
        "_chunk_health",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("health fault")),
    )

    with pytest.raises(OSError, match="health fault"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="OSError")
    assert not manifest_path.exists()


def test_manifest_write_exception_finalization_fails_closed(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    atomic_text = subject._atomic_text

    def fail_manifest_write(path: Path, text: str) -> None:
        if Path(path) == manifest_path:
            raise PermissionError("manifest write fault")
        atomic_text(path, text)

    monkeypatch.setattr(subject, "_atomic_text", fail_manifest_write)

    with pytest.raises(PermissionError, match="manifest write fault"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="PermissionError")
    assert not manifest_path.exists()


def test_manifest_live_verify_exception_removes_pseudo_complete(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    verify_manifest = subject._verify_manifest
    calls = 0

    def fail_live_verify(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("live manifest verify fault")
        verify_manifest(*args, **kwargs)

    monkeypatch.setattr(subject, "_verify_manifest", fail_live_verify)

    with pytest.raises(RuntimeError, match="live manifest verify fault"):
        subject.run_pilot(tmp_path)

    assert calls == 2
    _assert_failed_heartbeat(heartbeat_path, error_type="RuntimeError")
    assert not manifest_path.exists()


def test_complete_heartbeat_exception_finalization_fails_closed(
    tmp_path, monkeypatch
) -> None:
    _cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )
    heartbeat = subject._heartbeat

    def fail_complete_heartbeat(*args, **kwargs) -> None:
        if kwargs.get("state") == "COMPLETE":
            raise OSError("complete heartbeat fault")
        heartbeat(*args, **kwargs)

    monkeypatch.setattr(subject, "_heartbeat", fail_complete_heartbeat)

    with pytest.raises(OSError, match="complete heartbeat fault"):
        subject.run_pilot(tmp_path)

    _assert_failed_heartbeat(heartbeat_path, error_type="OSError")
    assert not manifest_path.exists()


def test_healthy_finalization_commits_manifest_before_complete(
    tmp_path, monkeypatch
) -> None:
    cells, manifest_path, heartbeat_path = _install_run_pilot_fixture(
        tmp_path, monkeypatch
    )

    report = subject.run_pilot(tmp_path)

    assert report == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["status"] == subject.STATUS
    assert report["observed_cells"] == len(cells)
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["completed_cells"] == len(cells)
    assert heartbeat["active"] is False
    assert heartbeat["state"] == "COMPLETE"
    assert heartbeat["error_type"] is None
    assert heartbeat["manifest"]["path"] == manifest_path.name
    assert heartbeat["manifest_analysis_sha256"] == report["analysis_sha256"]

    heartbeat_path.unlink()
    resumed = subject.run_pilot(tmp_path)
    repaired = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert resumed == report
    assert repaired["state"] == "COMPLETE"
    assert repaired["manifest_analysis_sha256"] == report["analysis_sha256"]


def test_direct_unverified_pilot_import_is_not_an_execution_entrypoint() -> None:
    with pytest.raises(
        RuntimeError,
        match="trusted-operator bootstrap",
    ):
        subject._require_verified_self_import()


def test_stdlib_bootstrap_rejects_self_restoring_pilot_swap(
    tmp_path,
) -> None:
    from cnn_fpga.benchmark import phase9_high_cutoff_design_bootstrap as bootstrap

    with pytest.raises(RuntimeError, match="trusted-operator isolated launcher"):
        bootstrap.main(["probe"])

    module_name = "phase9_bootstrap_self_restoring_attack_fixture"
    source_path = tmp_path / "attack.py"
    trusted_source = "EVIL = False\n"
    trusted_sha256 = sha256(trusted_source.encode("utf-8")).hexdigest()
    attacker_source = (
        "from pathlib import Path\n"
        "EVIL = True\n"
        f"Path(__file__).write_text({trusted_source!r}, encoding='utf-8')\n"
    )
    source_path.write_text(attacker_source, encoding="utf-8")

    with pytest.raises(RuntimeError, match="source byte drift"):
        bootstrap._load_verified_module(
            tmp_path,
            module_name,
            "attack.py",
            trusted_sha256,
        )

    assert module_name not in sys.modules
    assert source_path.read_text(encoding="utf-8") == attacker_source


def test_bootstrap_independent_sha256_matches_known_vectors() -> None:
    from cnn_fpga.benchmark import phase9_high_cutoff_design_bootstrap as bootstrap

    for payload in (b"", b"abc", bytes(range(256)), b"phase9" * 1000):
        assert bootstrap._sha256_bytes(payload) == sha256(payload).hexdigest()


def test_external_launcher_rejects_live_hashed_malicious_bootstrap(tmp_path) -> None:
    bootstrap_path = (
        tmp_path / "cnn_fpga/benchmark/phase9_high_cutoff_design_bootstrap.py"
    )
    bootstrap_path.parent.mkdir(parents=True)
    bootstrap_path.write_text(
        "print('{\"attack_accepted\": true}')\n",
        encoding="utf-8",
    )

    with pytest.raises(subprocess.CalledProcessError):
        _external_bootstrap_probe(tmp_path)


def test_bootstrap_rejects_unbound_launcher_with_preloaded_fake_hashlib() -> None:
    malicious_launcher = (
        "import pathlib,sys,types\n"
        "class FakeDigest:\n"
        " def hexdigest(self): return sys.argv[2]\n"
        "fake=types.ModuleType('hashlib')\n"
        "fake.sha256=lambda payload=b'': FakeDigest()\n"
        "sys.modules['hashlib']=fake\n"
        "root=pathlib.Path(sys.argv[1]).resolve()\n"
        "path=root/'cnn_fpga/benchmark/phase9_high_cutoff_design_bootstrap.py'\n"
        "payload=path.read_bytes()\n"
        "binding={'path':path.relative_to(root).as_posix(),"
        "'bytes':len(payload),'sha256':sys.argv[2]}\n"
        "namespace={'__name__':'__main__','__file__':str(path),"
        "'__package__':'cnn_fpga.benchmark',"
        "'__verified_source_sha256__':sys.argv[2],"
        "'__verified_external_launcher_sha256__':sys.argv[3],"
        "'__verified_external_launcher_source__':sys.orig_argv[4],"
        "'__verified_external_launcher_flags__':('-I','-S'),"
        "'__verified_bootstrap_source_binding__':binding}\n"
        "sys.argv=[str(path),'probe']\n"
        "exec(compile(payload,str(path),'exec',dont_inherit=True),namespace)\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            malicious_launcher,
            str(ROOT),
            PREREGISTERED_BOOTSTRAP_SHA256,
            PREREGISTERED_EXTERNAL_LAUNCHER_SHA256,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0
    assert "MALICIOUS_PILOT_EXECUTED_THROUGH_FIXED_BOOTSTRAP" not in completed.stdout
    assert "trusted-operator isolated launcher" in completed.stderr


def test_stdlib_bootstrap_attests_pilot_and_same_process_worker_chain() -> None:
    from cnn_fpga.benchmark import phase9_high_cutoff_design_bootstrap as bootstrap

    assert bootstrap.EXTERNAL_LAUNCHER_SOURCE == PREREGISTERED_EXTERNAL_LAUNCHER_SOURCE
    assert bootstrap.EXTERNAL_LAUNCHER_SHA256 == PREREGISTERED_EXTERNAL_LAUNCHER_SHA256
    assert (
        sha256(PREREGISTERED_EXTERNAL_LAUNCHER_SOURCE.encode("utf-8")).hexdigest()
        == PREREGISTERED_EXTERNAL_LAUNCHER_SHA256
    )
    assert (
        sha256(Path(bootstrap.__file__).read_bytes()).hexdigest()
        == PREREGISTERED_BOOTSTRAP_SHA256
    )
    attestation = _external_bootstrap_probe()

    assert attestation["pilot_sha256"] == bootstrap.PILOT_SHA256
    assert attestation["bootstrap_contract"] == bootstrap.VERIFIED_LOADER_CONTRACT
    assert attestation["execution_module_count"] == len(subject._EXECUTION_MODULE_NAMES)
    assert (
        attestation["fresh_runner_sha256"]
        == attestation["expected_fresh_runner_sha256"]
    )
    assert len(str(attestation["external_launcher_sha256"])) == 64
    assert attestation["bootstrap_sha256"] == PREREGISTERED_BOOTSTRAP_SHA256
    assert attestation["launcher_assurance"] == bootstrap.LAUNCHER_ASSURANCE
    assert (
        attestation["launcher_assurance"]["cryptographic_process_origin_attestation"]
        is None
    )
    assert (
        attestation["launcher_assurance"]["adversarial_local_operator_resistance"]
        is None
    )
    assert attestation["thread_worker_attestations"] == 6
