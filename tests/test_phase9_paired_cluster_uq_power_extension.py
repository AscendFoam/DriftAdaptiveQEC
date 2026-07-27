from __future__ import annotations

import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_paired_cluster_uq_power_extension as subject


ROOT = Path(__file__).resolve().parents[1]


def test_live_extension_preserves_parent_no_go_and_gates() -> None:
    extension, parent_config, parent_report = subject.load_extension(ROOT)
    child = subject.materialize_child_config(extension, parent_config)
    assert parent_report["verdict"] == "NO_GO_PAIRED_CLUSTER_UQ_CALIBRATION"
    assert parent_report["selected_formal_clusters_per_state"] is None
    assert child["cluster_counts_per_state"] == [384, 512]
    assert child["candidate_calibration_factors"] == [1.0]
    assert child["selection_gates"] == parent_config["selection_gates"]
    assert child["splits"] == extension["splits"]
    assert extension["claim_boundary"]["twin_qualification"] is None


def test_child_materialization_does_not_mutate_parent() -> None:
    extension, parent_config, _ = subject.load_extension(ROOT)
    before = json.dumps(parent_config, sort_keys=True)
    subject.materialize_child_config(extension, parent_config)
    assert json.dumps(parent_config, sort_keys=True) == before


def test_seed_firewall_rejects_any_cross_namespace_collision() -> None:
    calibration = [{"trial_seed": 1, "multiplier_seed": 3}]
    validation = [{"trial_seed": 2, "multiplier_seed": 1}]
    with pytest.raises(RuntimeError, match="seed collision"):
        subject._seed_firewall(calibration, validation)


def test_help_is_zero_write(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(subject, "write_artifacts", lambda: calls.append(True))
    with pytest.raises(SystemExit) as raised:
        subject.main(["--help"])
    assert raised.value.code == 0
    assert calls == []
