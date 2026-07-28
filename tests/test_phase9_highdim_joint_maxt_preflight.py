from __future__ import annotations

from dataclasses import replace
import ast
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_highdim_joint_maxt_preflight as writer
from cnn_fpga.benchmark import phase9_highdim_joint_maxt_preflight_verify as verifier


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    config = json.loads((ROOT / writer.CONFIG_PATH).read_text(encoding="utf-8"))
    old = json.loads(
        (ROOT / config["parent_contracts"]["old_seven_family_design"]).read_text(
            encoding="utf-8"
        )
    )
    repair = json.loads(
        (ROOT / config["parent_contracts"]["bounded_repair"]).read_text(
            encoding="utf-8"
        )
    )
    return config, old, repair


def test_exact_blueprint_is_independently_reproducible():
    config, old, repair = _inputs()
    primary = [writer.asdict(gate) for gate in writer.build_blueprint(config, old, repair)]
    independent = verifier.expected_blueprint(config, old, repair)
    assert primary == independent
    assert len(primary) == 3043
    assert len({row["gate_id"] for row in primary}) == 3043
    assert sum(row["source_contract"] == "old_seven_family" for row in primary) == 1589
    assert sum(row["source_contract"] == "t03_bounded_repair" for row in primary) == 1454
    assert sum(not row["deterministic"] for row in primary) == 3037
    assert {row["cluster_count"] for row in primary} == {0, 384, 768, 2304}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("density_uq", "dimensions"), [120]),
        (("density_uq", "clusters_per_state"), 383),
        (("density_uq", "multiplier_replicates"), 198),
        (("density_uq", "calibration_factor"), 1.01),
        (("density_uq", "quantile_method"), "linear"),
        (("joint_maxt", "aggregate_rescue"), True),
        (("joint_maxt", "gate_deletion"), True),
        (("joint_maxt", "cross_state_averaging"), True),
        (("joint_maxt", "pointwise_z_substitution"), True),
        (("formal_outcome_access", "t04_formal_outcomes_accessed"), True),
    ],
)
def test_frozen_contract_mutations_fail_closed(path, value):
    config, _, _ = _inputs()
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        writer._validate_config(config)


def test_seed_address_is_injective_and_ranges_are_disjoint():
    config, _, _ = _inputs()
    cells = writer._density_cells(config)
    trials = config["density_uq"]["trial_count_per_cell"]
    trial = {
        config["density_uq"]["trial_seed_base"] + cell["cell_index"] * trials + i
        for cell in cells for i in range(trials)
    }
    multiplier = {
        config["density_uq"]["multiplier_seed_base"]
        + cell["cell_index"] * trials + i
        for cell in cells for i in range(trials)
    }
    assert len(trial) == len(cells) * trials
    assert len(multiplier) == len(cells) * trials
    assert trial.isdisjoint(multiplier)


def test_one_real_d132_density_trial_matches_independent_implementation():
    config, _, _ = _inputs()
    family = config["density_uq"]["families"]["heavy_tail_rare_coherent"]
    seed, multiplier_seed = 991337, 991338
    left, right, truth = writer._physical_density_trial(
        dimension=132, count=12, true_distance=0.05, family=family, seed=seed
    )
    left_i, right_i = verifier._density_trial(132, 12, 0.05, family, seed)
    assert truth == pytest.approx(0.05, abs=1e-14)
    assert np.max(np.abs(left - left_i)) < 2e-14
    assert np.max(np.abs(right - right_i)) < 2e-14
    observed = writer.paired_density_trace_ucb_physical(
        left, right, confidence=0.95, multiplier_replicates=199,
        seed=multiplier_seed, calibration_factor=1.0,
    )
    estimate, radius, upper = verifier._density_ucb(
        left_i, right_i, confidence=0.95, b=199, seed=multiplier_seed
    )
    assert observed.estimate == pytest.approx(estimate, abs=2e-13)
    assert observed.raw_radius == pytest.approx(radius, abs=2e-13)
    assert observed.upper_bound == pytest.approx(upper, abs=2e-13)


def test_full_maxt_is_reproducible_and_pointwise_z_is_not_used():
    config, old, repair = _inputs()
    gates = writer.build_blueprint(config, old, repair)
    q1, raw1, power1, _ = writer.build_joint_maxt(config, gates)
    q2, maxima, power2 = verifier._recompute_maxt(
        config, [writer.asdict(gate) for gate in gates]
    )
    assert q1 == pytest.approx(q2, abs=2e-12)
    assert q1 > 1.6448536269514722
    recorded = [row["max_statistic"] for row in raw1 if row["row_type"] == "maxt_replicate"]
    assert recorded == pytest.approx(maxima, abs=2e-12)
    assert {
        (row["case"], row["family"], row["successes"]) for row in power1
    } == {
        (row["case"], row["family"], row["successes"]) for row in power2
    }


def test_verifier_has_no_prohibited_project_imports():
    tree = ast.parse(
        (ROOT / "cnn_fpga/benchmark/phase9_highdim_joint_maxt_preflight_verify.py")
        .read_text(encoding="utf-8")
    )
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith(("physics", "cnn_fpga")) for name in imported)


def test_scale_two_is_a_strict_power_improvement_without_gate_deletion():
    """Regression for the observed base-count joint-maxT power bottleneck.

    This is diagnostic only: T05 must report the base-count NO-GO rather than
    silently applying this scale or deleting hard gates.
    """
    config, old, repair = _inputs()
    gates = writer.build_blueprint(config, old, repair)
    _, _, _, base_decisions = writer.build_joint_maxt(config, gates)
    doubled = [
        replace(
            gate,
            cluster_count=(
                2 * gate.cluster_count if not gate.deterministic else 0
            ),
        )
        for gate in gates
    ]
    _, _, _, doubled_decisions = writer.build_joint_maxt(config, doubled)
    assert not all(base_decisions.values())
    assert all(doubled_decisions.values())
    assert len(doubled) == len(gates) == 3043


def test_cli_help_is_side_effect_free(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        target = Path(directory)
        monkeypatch.setattr(writer, "_root", lambda: target)
        with pytest.raises(SystemExit) as exc:
            writer.main(["--help"])
        assert exc.value.code == 0
        assert list(target.rglob("*")) == []
