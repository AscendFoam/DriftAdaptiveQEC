from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import phase9_count_selection_confirmation_verify as subject


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / subject.CONFIG_PATH).read_text(encoding="utf-8"))


def _blueprint(config: dict) -> dict:
    return json.loads(
        (ROOT / config["parent_artifacts"]["t05_blueprint"]).read_text(
            encoding="utf-8"
        )
    )


def _set(config: dict, path: tuple[str, ...], value: object) -> dict:
    mutated = copy.deepcopy(config)
    target = mutated
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    return mutated


def test_live_config_and_verifier_import_independence() -> None:
    subject.validate_config(_config())
    assert subject._validate_import_independence()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("density", "factor"), 1.01),
        (("density", "multiplier_replicates"), 99),
        (("density", "quantile"), "linear"),
        (("joint_maxt", "factor"), 1.01),
        (("joint_maxt", "multiplier_replicates"), 999),
        (("joint_maxt", "quantile"), "linear"),
        (("joint_maxt", "pointwise_z_substitution"), True),
        (("joint_maxt", "gate_deletion"), True),
        (("outcome_firewall", "t04_formal_outcomes_accessed"), True),
        (("claim_boundary", "official_puviani_exact"), "PASS"),
    ],
)
def test_factor_b_quantile_pointwise_outcome_and_claim_mutations_fail(
    path: tuple[str, ...], value: object
) -> None:
    with pytest.raises(ValueError):
        subject.validate_config(_set(_config(), path, value))


def test_posthoc_linked_scale_mutation_fails() -> None:
    config = _config()
    config["linked_count_grid"][1]["scale"] = 2.1
    with pytest.raises(ValueError, match="linked count grid"):
        subject.validate_config(config)


def test_seed_namespace_reuse_fails() -> None:
    config = _config()
    config["density"]["confirmation_trial_seed_base"] = config["density"][
        "selection_trial_seed_base"
    ]
    with pytest.raises(ValueError, match="seed"):
        subject.validate_config(config)


def test_deleted_or_duplicate_blueprint_gate_fails_even_if_resigned() -> None:
    config = _config()
    blueprint = _blueprint(config)
    deleted = copy.deepcopy(blueprint)
    deleted["gates"].pop()
    deleted["analysis_sha256"] = subject._self_hash(deleted)
    with pytest.raises(ValueError, match="denominator"):
        subject.scale_blueprint(deleted, config["linked_count_grid"][0])

    duplicate = copy.deepcopy(blueprint)
    duplicate["gates"][-1] = copy.deepcopy(duplicate["gates"][0])
    duplicate["analysis_sha256"] = subject._self_hash(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        subject.scale_blueprint(duplicate, config["linked_count_grid"][0])


def test_all_linked_blueprints_preserve_3043_3037_and_counts() -> None:
    config = _config()
    blueprint = _blueprint(config)
    for candidate in config["linked_count_grid"]:
        gates = subject.scale_blueprint(blueprint, candidate)
        assert len(gates) == 3043
        assert sum(not gate["deterministic"] for gate in gates) == 3037
        assert {gate["cluster_count"] for gate in gates} == {
            0,
            candidate["state_clusters"],
            candidate["round_clusters"],
            candidate["aggregate_fault_clusters"],
        }


def test_smallest_candidate_selection_cannot_skip_an_earlier_pass() -> None:
    config = _config()
    density = {1.5: False, 2.0: True, 2.5: True, 3.0: True}
    maxt = {1.5: True, 2.0: True, 2.5: True, 3.0: True}
    assert subject.select_smallest(config, density, maxt) == config[
        "linked_count_grid"
    ][1]
    assert subject.select_smallest(
        config,
        {scale: False for scale in density},
        maxt,
    ) is None


def test_density_trial_is_complex_psd_and_b199_higher_is_reproducible() -> None:
    family = {
        "spectrum_profile": "low_energy",
        "left_noise_weight": 0.25,
        "right_noise_weight": 0.75,
        "rare_probability": 1.0,
        "coherent_unitary": True,
    }
    left, right, truth = subject._density_trial(
        dimension=8, count=6, truth=0.05, family=family, seed=123
    )
    assert np.iscomplexobj(left)
    assert np.iscomplexobj(right)
    assert truth == pytest.approx(0.05, abs=1e-12)
    first = subject._density_ucb(
        left, right, confidence=0.95, replicates=199, seed=456
    )
    second = subject._density_ucb(
        left, right, confidence=0.95, replicates=199, seed=456
    )
    assert first == second
    assert first[2] == pytest.approx(first[0] + first[1])


def test_non_psd_density_is_rejected() -> None:
    invalid = np.zeros((2, 4, 4), dtype=np.complex128)
    invalid[:, 0, 0] = 1.1
    invalid[:, 1, 1] = -0.1
    with pytest.raises(ValueError, match="non-physical"):
        subject._validate_density_stack(invalid, "mutation")


def test_summary_tamper_is_detected() -> None:
    expected = [{
        "split": "selection",
        "candidate_scale": 1.5,
        "cluster_count": 576,
        "dimension": 120,
        "trials": 128,
        "coverage_successes": 128,
        "coverage_lcb": 0.9,
        "equivalence_successes": 100,
        "equivalence_rate": 100 / 128,
        "equivalence_lcb": 0.7,
        "equivalence_ucb": 0.8,
        "gate_pass": True,
    }]
    tampered = copy.deepcopy(expected)
    tampered[0]["equivalence_successes"] = 101
    with pytest.raises(ValueError, match="mismatch"):
        subject._compare_density_summaries(
            expected, tampered, split="selection"
        )


def test_source_binding_tamper_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_text("row_type,value\nx,1\n", encoding="utf-8")
    binding = subject._binding(path, tmp_path)
    path.write_text("row_type,value\nx,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        subject._safe_bound_path(tmp_path, binding)


def test_resource_algebra_uses_selected_linked_scale_and_keeps_fresh_gate() -> None:
    config = _config()
    selected = config["linked_count_grid"][1]
    free = 300_000_000_000
    summary, passed = subject._resource_summary(config, selected, free)
    assert summary["exact_t04_rows"] == 2 * config["resource"]["base_t04_rows"]
    assert summary["fresh_t04_layer_benchmark_required"] is True
    assert passed


def test_report_claim_population_is_rejected() -> None:
    report = {
        "qualified_claim": None,
        "claim_boundary": {
            **{name: None for name in subject.NULL_CLAIMS},
            "count_design_only": True,
            "t04_scientific_execution_released": False,
        },
    }
    subject._check_report_claims(report)
    report["claim_boundary"]["ler"] = 0.01
    with pytest.raises(ValueError, match="claim"):
        subject._check_report_claims(report)


@pytest.mark.parametrize(
    ("family_name", "dimension", "count", "effect", "trial_seed", "multiplier_seed"),
    [
        ("heteroskedastic_coherent", 120, 576, 0.05, 201_000_003, 202_000_003),
        ("heavy_tail_rare_coherent", 132, 768, 0.12, 203_000_017, 204_000_017),
    ],
)
def test_independent_density_matches_frozen_production_reference(
    family_name: str,
    dimension: int,
    count: int,
    effect: float,
    trial_seed: int,
    multiplier_seed: int,
) -> None:
    from cnn_fpga.benchmark import phase9_highdim_joint_maxt_preflight as t05

    config = _config()
    t05_config = json.loads(
        (ROOT / config["parent_artifacts"]["t05_config"]).read_text(
            encoding="utf-8"
        )
    )
    family = t05_config["density_uq"]["families"][family_name]
    expected_left, expected_right, expected_truth = t05._physical_density_trial(
        dimension=dimension,
        count=count,
        true_distance=effect,
        family=family,
        seed=trial_seed,
    )
    actual_left, actual_right, actual_truth = subject._density_trial(
        dimension=dimension,
        count=count,
        truth=effect,
        family=family,
        seed=trial_seed,
    )
    assert actual_truth == pytest.approx(expected_truth, abs=1e-14)
    assert np.array_equal(actual_left, expected_left)
    assert np.array_equal(actual_right, expected_right)

    expected_ucb = t05.paired_density_trace_ucb_physical(
        expected_left,
        expected_right,
        confidence=0.95,
        multiplier_replicates=199,
        seed=multiplier_seed,
        calibration_factor=1.0,
    )
    actual_ucb = subject._density_ucb(
        actual_left,
        actual_right,
        confidence=0.95,
        replicates=199,
        seed=multiplier_seed,
    )
    assert actual_ucb == pytest.approx(
        (
            expected_ucb.estimate,
            expected_ucb.raw_radius,
            expected_ucb.upper_bound,
        ),
        abs=2e-14,
    )


def test_independent_full_blueprint_maxt_matches_frozen_reference() -> None:
    from cnn_fpga.benchmark import phase9_count_selection_confirmation as writer
    from cnn_fpga.benchmark import phase9_highdim_joint_maxt_preflight as t05

    config = _config()
    t05_config = json.loads(
        (ROOT / config["parent_artifacts"]["t05_config"]).read_text(
            encoding="utf-8"
        )
    )
    blueprint = _blueprint(config)
    candidate = config["linked_count_grid"][0]
    expected_config = writer._maxt_config(
        t05_config, config, split="selection"
    )
    expected_gates = writer._scaled_gates(blueprint["gates"], candidate)
    expected_critical, expected_raw, expected_power, expected_decisions = (
        t05.build_joint_maxt(expected_config, expected_gates)
    )
    actual_critical, actual_maxima, actual_power, actual_decisions = (
        subject.recompute_maxt(
            config,
            subject.scale_blueprint(blueprint, candidate),
            split="selection",
        )
    )

    expected_maxima = [
        row["max_statistic"]
        for row in expected_raw
        if row["row_type"] == "maxt_replicate"
    ]
    assert actual_critical == pytest.approx(expected_critical, abs=2e-14)
    assert actual_maxima == pytest.approx(expected_maxima, abs=2e-14)
    assert actual_decisions == expected_decisions
    assert len(actual_power) == len(expected_power) == 40
    by_key = {
        (row["case"], row["family"]): row for row in expected_power
    }
    for row in actual_power:
        expected = by_key[(row["case"], row["family"])]
        assert row["successes"] == expected["successes"]
        assert row.get("injection_gate_id") == expected.get("injection_gate_id")
        for name in ("point", "wilson_lcb", "wilson_ucb"):
            assert row[name] == pytest.approx(expected[name], abs=2e-14)
