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


@pytest.mark.parametrize(
    ("computed", "expected"),
    [
        (0.0, 0.0),
        (0.04999999999999993, 0.05),
        (0.05000000000000002, 0.05),
        (0.10000000000000003, 0.1),
        (0.11999999999999997, 0.12),
    ],
)
def test_registered_effect_maps_roundoff_to_frozen_label(computed, expected):
    assert subject._registered_effect(
        computed, [0.0, 0.05, 0.1, 0.12]
    ) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.051, -0.01])
def test_registered_effect_rejects_nonfinite_or_unregistered_value(value):
    with pytest.raises(ValueError):
        subject._registered_effect(value, [0.0, 0.05, 0.1, 0.12])


def _confirmation_rows(effect, computed, *, equivalent):
    cell_id = (
        "confirmation__n768__d120__heteroskedastic_coherent__"
        f"effect_{effect:.3f}"
    )
    return [
        {
            "cell_id": cell_id,
            "split": "confirmation",
            "candidate_scale": 2.0,
            "family": "heteroskedastic_coherent",
            "dimension": 120,
            "true_distance": computed,
            "cluster_count": 768,
            "trial": trial,
            "covered": True,
            "equivalence_pass": equivalent,
        }
        for trial in range(256)
    ]


@pytest.mark.parametrize(
    ("effect", "computed", "equivalent"),
    [
        (0.0, 0.0, True),
        (0.05, 0.05000000000000002, True),
        (0.1, 0.10000000000000003, False),
        (0.12, 0.11999999999999997, False),
    ],
)
def test_confirmation_dispatches_all_four_registered_power_rules(
    effect, computed, equivalent
):
    config, _, _ = _load()
    summaries, passed = subject._summarize_confirmation(
        config,
        _confirmation_rows(effect, computed, equivalent=equivalent),
        require_complete=False,
    )
    assert passed is True
    assert len(summaries) == 1
    assert summaries[0]["true_distance"] == computed
    assert summaries[0]["coverage_gate_pass"] is True
    assert summaries[0]["power_gate_pass"] is True
    assert summaries[0]["gate_pass"] is True


def test_confirmation_rejects_frozen_label_and_computed_effect_disagreement():
    config, _, _ = _load()
    rows = _confirmation_rows(0.05, 0.1, equivalent=True)
    with pytest.raises(ValueError, match="disagrees with frozen label"):
        subject._summarize_confirmation(config, rows, require_complete=False)


def test_confirmation_rejects_unregistered_or_ambiguous_cell_label():
    config, _, _ = _load()
    rows = _confirmation_rows(0.05, 0.05, equivalent=True)
    for row in rows:
        row["cell_id"] = (
            "confirmation__n768__d120__heteroskedastic_coherent__effect_0.051"
        )
    with pytest.raises(ValueError, match="no unique registered effect label"):
        subject._summarize_confirmation(config, rows, require_complete=False)


def test_confirmation_rejects_mixed_computed_effects_within_cell():
    config, _, _ = _load()
    rows = _confirmation_rows(0.05, 0.05, equivalent=True)
    rows[-1]["true_distance"] = 0.1
    with pytest.raises(ValueError, match="disagrees with frozen label"):
        subject._summarize_confirmation(config, rows, require_complete=False)


def test_confirmation_requires_complete_frozen_cartesian_product():
    config, _, _ = _load()
    rows = _confirmation_rows(0.05, 0.05, equivalent=True)
    with pytest.raises(ValueError, match="Cartesian product drift"):
        subject._summarize_confirmation(config, rows)


def test_reuse_only_rejects_missing_chunks_without_creating_directory():
    config, t05_config, _ = _load()
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        target = Path(directory)
        with pytest.raises(RuntimeError, match="reuse-only chunk directory"):
            subject._run_density(
                target,
                config,
                t05_config,
                split="selection",
                workers=4,
                reuse_only=True,
            )
        assert not (
            target / config["artifact_paths"]["selection_chunks"]
        ).exists()


def test_help_path_writes_nothing(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        target = Path(directory)
        monkeypatch.setattr(subject, "_root", lambda: target)
        with pytest.raises(SystemExit) as exc:
            subject.main(["--help"])
        assert exc.value.code == 0
        assert list(target.rglob("*")) == []
