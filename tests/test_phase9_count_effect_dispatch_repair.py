from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile

import pytest

from cnn_fpga.benchmark import phase9_count_effect_dispatch_repair as subject


ROOT = Path(__file__).resolve().parents[1]


def _load():
    config = json.loads((ROOT / subject.CONFIG_PATH).read_text(encoding="utf-8"))
    t06_config = json.loads(
        (ROOT / config["artifact_paths"]["t06_config"]).read_text(
            encoding="utf-8"
        )
    )
    return config, t06_config


def test_repair_config_archive_and_source_seals_validate():
    config, _ = _load()
    subject.validate_config(config, ROOT)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("repair_contract", "new_random_trials", True),
        ("repair_contract", "raw_chunk_rewrite", True),
        ("repair_contract", "absolute_tolerance", 1e-3),
        ("repair_contract", "factor_change", True),
        ("claim_boundary", "t04_preregistration_final_release", True),
        ("claim_boundary", "official_puviani_exact", True),
    ],
)
def test_repair_scope_mutations_fail_closed(section, key, value):
    config, _ = _load()
    config[section][key] = value
    with pytest.raises(ValueError):
        subject.validate_config(config, ROOT)


def test_runtime_contract_mutation_fails_closed():
    config, _ = _load()
    config["runtime_contract"][
        "expected_single_thread_maxT_string_differences_vs_v1"
    ] = 1
    with pytest.raises(ValueError, match="runtime contract drift"):
        subject.validate_config(config, ROOT)


def test_runtime_environment_mismatch_fails_before_writer(monkeypatch):
    config, _ = _load()
    required = config["runtime_contract"][
        "required_process_environment_before_python_start"
    ]
    for name, value in required.items():
        monkeypatch.setenv(name, value)
    subject._validate_runtime_environment(config)

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    called = False

    def forbidden_writer(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("writer must not be called")

    monkeypatch.setattr(subject.writer, "write_artifacts", forbidden_writer)
    with pytest.raises(RuntimeError, match="environment not frozen"):
        subject.repair(ROOT)
    assert called is False


def test_full_v1_archive_manifest_and_chunk_inventory_are_exact():
    config, t06_config = _load()
    archive = ROOT / config["v1_archive"]["local_run_archive"]
    entries, digest = subject._full_archive_manifest(archive)
    assert len(entries) == 38
    assert sum(entry["bytes"] for entry in entries) == 7_548_008
    assert digest == "b92a8fe4b3e0d589e3642d4dee315edad341c3b09267dbc11db5c6c20cf09b57"

    inventory, rows = subject._chunk_inventory(ROOT, config, t06_config)
    assert len(inventory) == 32
    assert sum(item["split"] == "selection" for item in inventory) == 8
    assert sum(item["split"] == "confirmation" for item in inventory) == 24
    assert len(rows) == 7168
    assert len({row["trial_seed"] for row in rows}) == 7168
    subject._verify_csv_raw_matches_chunks(
        ROOT / config["v1_archive"]["source_data"]["path"],
        rows,
    )
    summaries, passed = subject.writer._summarize_confirmation(
        t06_config,
        [row for row in rows if row["split"] == "confirmation"],
    )
    assert len(summaries) == 24
    assert passed is True
    assert all(row["gate_pass"] for row in summaries)


def test_archived_v1_signature_is_the_exact_float_dispatch_failure():
    config, _ = _load()
    report = json.loads(
        (ROOT / config["v1_archive"]["writer_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    subject._validate_v1_signature(report)
    failed = [
        row
        for row in report["confirmation"]["density_summaries"]
        if not row["gate_pass"]
    ]
    assert len(failed) == 6
    assert all(row["cell_id"].endswith("__effect_0.050") for row in failed)
    assert all(row["equivalence_successes"] == 256 for row in failed)


def _synthetic_repaired_report(config):
    report = json.loads(
        (ROOT / config["v1_archive"]["writer_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    repaired = copy.deepcopy(report)
    changed_ids = set(
        config["expected_invariants"]["permitted_changed_cell_ids"]
    )
    for row in repaired["confirmation"]["density_summaries"]:
        if row["cell_id"] in changed_ids:
            row["power_gate_pass"] = True
            row["gate_pass"] = True
    repaired["bindings"]["source_data"]["bytes"] -= 12
    repaired["bindings"]["source_data"]["sha256"] = "1" * 64
    repaired["bindings"]["writer_source"]["bytes"] += 100
    repaired["bindings"]["writer_source"]["sha256"] = "2" * 64
    repaired["gates"]["G10_density_confirmation_all_pass"] = True
    repaired["gate_summary"]["passed"] = 15
    repaired["t04_preregistration_released"] = True
    repaired["claim_boundary"]["t04_preregistration_released"] = True
    repaired["verdict"] = subject.EXPECTED_V2_VERDICT
    repaired["analysis_sha256"] = "3" * 64
    repaired["resource"]["free_disk_bytes"] -= 1
    return report, repaired


def test_writer_report_diff_whitelist_is_exact_and_rejects_scientific_drift():
    config, _ = _load()
    v1, v2 = _synthetic_repaired_report(config)
    changed = subject._compare_writer_reports(v1, v2, config)
    assert "gates.G10_density_confirmation_all_pass" in changed
    assert "resource.free_disk_bytes" in changed
    assert len([
        path for path in changed if path.endswith(".power_gate_pass")
    ]) == 6
    assert len([
        path for path in changed if path.endswith(".gate_pass")
    ]) == 6

    v2["selection"]["selected"]["scale"] = 2.5
    with pytest.raises(ValueError, match="diff whitelist mismatch"):
        subject._compare_writer_reports(v1, v2, config)


def _write_csv(path, rows):
    fields = [
        "cell_id",
        "row_type",
        "true_distance",
        "power_gate_pass",
        "gate_pass",
        "estimate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_source_diff_allows_only_two_booleans_in_registered_local_cell(tmp_path):
    config, _ = _load()
    config = copy.deepcopy(config)
    config["expected_invariants"]["source_row_count"] = 2
    config["expected_invariants"]["permitted_changed_summary_cells"] = 1
    config["expected_invariants"]["permitted_changed_cell_ids"] = [
        "confirmation__n768__d120__x__effect_0.050"
    ]
    old = [
        {
            "cell_id": "confirmation__n768__d120__x__effect_0.050",
            "row_type": "",
            "true_distance": "0.05000000000000002",
            "power_gate_pass": "False",
            "gate_pass": "False",
            "estimate": "",
        },
        {
            "cell_id": "confirmation__n768__d120__x__effect_0.050",
            "row_type": "density_trial",
            "true_distance": "0.05000000000000002",
            "power_gate_pass": "",
            "gate_pass": "",
            "estimate": "0.04",
        },
    ]
    new = copy.deepcopy(old)
    new[0]["power_gate_pass"] = "True"
    new[0]["gate_pass"] = "True"
    old_path, new_path = tmp_path / "old.csv", tmp_path / "new.csv"
    _write_csv(old_path, old)
    _write_csv(new_path, new)
    changed = subject._compare_source_data(old_path, new_path, config)
    assert changed == [{
        "row_index": 0,
        "cell_id": old[0]["cell_id"],
        "computed_true_distance": 0.05000000000000002,
        "changed_fields": ["gate_pass", "power_gate_pass"],
    }]


def test_source_diff_rejects_any_raw_trial_change(tmp_path):
    config, _ = _load()
    config = copy.deepcopy(config)
    config["expected_invariants"]["source_row_count"] = 2
    config["expected_invariants"]["permitted_changed_summary_cells"] = 1
    config["expected_invariants"]["permitted_changed_cell_ids"] = [
        "confirmation__n768__d120__x__effect_0.050"
    ]
    old = [
        {
            "cell_id": "confirmation__n768__d120__x__effect_0.050",
            "row_type": "",
            "true_distance": "0.05000000000000002",
            "power_gate_pass": "False",
            "gate_pass": "False",
            "estimate": "",
        },
        {
            "cell_id": "confirmation__n768__d120__x__effect_0.050",
            "row_type": "density_trial",
            "true_distance": "0.05000000000000002",
            "power_gate_pass": "",
            "gate_pass": "",
            "estimate": "0.04",
        },
    ]
    new = copy.deepcopy(old)
    new[0]["power_gate_pass"] = "True"
    new[0]["gate_pass"] = "True"
    new[1]["estimate"] = "0.03"
    old_path, new_path = tmp_path / "old.csv", tmp_path / "new.csv"
    _write_csv(old_path, old)
    _write_csv(new_path, new)
    with pytest.raises(ValueError, match="unpermitted source-data change"):
        subject._compare_source_data(old_path, new_path, config)


def test_help_path_writes_nothing(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        target = Path(directory)
        monkeypatch.setattr(subject, "_root", lambda: target)
        with pytest.raises(SystemExit) as exc:
            subject.main(["--help"])
        assert exc.value.code == 0
        assert list(target.rglob("*")) == []
