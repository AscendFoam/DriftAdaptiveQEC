from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import pytest

from cnn_fpga.benchmark import phase9_count_selection_confirmation as subject


ROOT = Path(__file__).resolve().parents[1]


def _load():
    config = json.loads((ROOT / subject.CONFIG_PATH).read_text(encoding="utf-8"))
    t05_config = json.loads(
        (ROOT / config["parent_artifacts"]["t05_config"]).read_text(encoding="utf-8")
    )
    blueprint = json.loads(
        (ROOT / config["parent_artifacts"]["t05_blueprint"]).read_text(
            encoding="utf-8"
        )
    )["gates"]
    return config, t05_config, blueprint


def test_finite_linked_grid_and_config_validate():
    config, _, _ = _load()
    subject.validate_config(config)
    assert [row["state_clusters"] for row in config["linked_count_grid"]] == [
        576, 768, 960, 1152
    ]
    assert config["outcome_firewall"]["candidate_exhaustion"].startswith("TERMINAL")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("density", "factor", 1.01),
        ("density", "multiplier_replicates", 99),
        ("density", "quantile", "linear"),
        ("joint_maxt", "gate_deletion", True),
        ("joint_maxt", "aggregate_rescue", True),
        ("joint_maxt", "pointwise_z_substitution", True),
        ("outcome_firewall", "t04_formal_outcomes_accessed", True),
        ("claim_boundary", "official_puviani_exact", True),
    ],
)
def test_contract_mutations_fail_closed(section, key, value):
    config, _, _ = _load()
    config[section][key] = value
    with pytest.raises(ValueError):
        subject.validate_config(config)


def test_scaled_blueprints_keep_all_gates_and_link_counts():
    config, _, blueprint = _load()
    for candidate in config["linked_count_grid"]:
        gates = subject._scaled_gates(blueprint, candidate)
        assert len(gates) == 3043
        assert len({gate.gate_id for gate in gates}) == 3043
        assert {gate.cluster_count for gate in gates} == {
            0,
            candidate["state_clusters"],
            candidate["round_clusters"],
            candidate["aggregate_fault_clusters"],
        }


def test_selection_and_confirmation_seed_ranges_are_disjoint_and_injective():
    config, t05_config, _ = _load()
    selection = subject._density_specs(config, t05_config, split="selection")
    confirmation = subject._density_specs(
        config, t05_config, split="confirmation",
        selected=config["linked_count_grid"][1],
    )
    sets = []
    for specs in (selection, confirmation):
        for kind in ("trial", "multiplier"):
            values = {
                spec[f"{kind}_seed_base"] + spec["cell_index"] * spec["trials"] + i
                for spec in specs for i in range(spec["trials"])
            }
            assert len(values) == sum(spec["trials"] for spec in specs)
            sets.append(values)
    for index, left in enumerate(sets):
        for right in sets[index + 1:]:
            assert left.isdisjoint(right)


def test_full_maxt_selection_evaluates_every_candidate_and_selects_two():
    config, t05_config, blueprint = _load()
    rows, decisions = subject._run_maxt_grid(config, t05_config, blueprint)
    summaries = [row for row in rows if row["row_type"] == "maxt_selection_summary"]
    assert len(summaries) == 4
    assert decisions == {1.5: False, 2.0: True, 2.5: True, 3.0: True}
    selected = next(
        row for row in config["linked_count_grid"] if decisions[row["scale"]]
    )
    assert selected["state_clusters"] == 768


def test_confirmation_cannot_run_without_selected_count():
    config, t05_config, _ = _load()
    with pytest.raises(ValueError):
        subject._density_specs(config, t05_config, split="confirmation")


def test_help_path_writes_nothing(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        target = Path(directory)
        monkeypatch.setattr(subject, "_root", lambda: target)
        with pytest.raises(SystemExit) as exc:
            subject.main(["--help"])
        assert exc.value.code == 0
        assert list(target.rglob("*")) == []
