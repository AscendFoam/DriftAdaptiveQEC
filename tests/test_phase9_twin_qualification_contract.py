from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_twin_qualification_contract as contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9/t9_2_4_twin_qualification.json"
SEAL = ROOT / "docs/t9_2_4_twin_qualification_preregistration.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path: Path, relative: str):
    payload = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def test_preregistration_rebuild_is_exact():
    assert contract.build_report(root=ROOT, config_path=CONFIG) == load(SEAL)


def test_live_verify_passes_without_formal_result_access():
    checks = contract.verify_report(root=ROOT, config_path=CONFIG)
    assert checks and all(checks.values())


def test_all_three_parents_are_hash_bound_and_claim_null():
    report = load(SEAL)
    expected = {
        "backend_a": "T9.2.2",
        "backend_b": "T9.2.3",
        "action_contract": "T9.2.1",
    }
    for key, task_id in expected.items():
        assert report["parents"][key]["task_id"] == task_id
        assert report["parents"][key]["all_outcome_claims_null"] is True
        assert len(report["parents"][key]["analysis_sha256"]) == 64


def test_config_and_implementation_are_live_bound():
    report = load(SEAL)
    assert report["config"]["path"] == (
        "configs/phase9/t9_2_4_twin_qualification.json"
    )
    assert report["config"]["bytes"] == CONFIG.stat().st_size
    assert report["config"]["sha256"] == sha256(CONFIG.read_bytes()).hexdigest()
    for value in report["implementation"].values():
        assert value == binding(ROOT / value["path"], value["path"])


def test_formal_seed_intervals_are_pairwise_disjoint():
    report = load(SEAL)
    intervals = [
        (value["start"], value["exclusive_stop"])
        for value in report["seed_intervals"].values()
    ]
    assert report["seed_intervals_pairwise_disjoint"] is True
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            assert left[1] <= right[0] or right[1] <= left[0]


def test_cell_accounting_is_exact_and_not_demo_sized():
    assert load(SEAL)["cell_accounting"] == {
        "mapping_cells": 2,
        "shared_state_action_rounds": 4480,
        "representative_probe_rounds": 512,
        "logical_state_action_rounds": 5376,
        "fault_trajectories": 384,
        "fault_trajectory_rounds": 4608,
        "total_formal_backend_rounds": 14976,
    }


def test_all_six_states_actions_probes_and_faults_are_frozen():
    config = load(CONFIG)
    assert config["formal_matrix"]["logical_labels"] == [
        "0",
        "1",
        "+",
        "-",
        "+i",
        "-i",
    ]
    assert len(config["formal_matrix"]["unique_nominal_actions"]) == 7
    assert len(config["action_contract"]["probe_ids"]) == 16
    assert set(config["formal_matrix"]["fault_scenarios"]) == {
        "step",
        "telegraph",
        "burst",
        "compound",
    }


def test_shared_physical_and_native_logical_layers_are_not_mixed():
    layers = load(SEAL)["comparison_layers"]
    assert "identical Fock/qutrit density" in layers["shared_physical_state"]
    assert "principal-angle mapping gate" in layers["native_logical_channel"]
    assert "independent backend seeds" in layers["fault_trajectory"]


def test_tolerances_are_prefrozen_before_formal_access():
    report = load(SEAL)
    assert report["formal_result_accessed"] is False
    assert len(report["prefrozen_tolerances"]) == 18
    assert report["prefrozen_tolerances"][
        "minimum_code_principal_singular_value"
    ] == 0.95
    assert report["prefrozen_tolerances"]["familywise_alpha"] == 0.05


def test_failure_policy_prohibits_rescue_and_downstream_lifetime():
    policy = load(SEAL)["failure_policy"]
    assert policy["fail_verdict"] == "NO_GO_TWIN_QUALIFICATION"
    assert "prohibit T9.2.5" in policy["main_metric_failure"]
    assert policy["no_retuning_after_formal_access"] is True
    assert policy["no_mean_only_rescue"] is True
    assert policy["no_cell_deletion"] is True
    assert policy["no_codebook_claim"] is True


def test_all_performance_and_external_claims_remain_null():
    state = load(SEAL)["current_null_state"]
    assert len(state) == 8
    assert all(value is None for value in state.values())
    assert state["official_puviani_exact"] is None
    assert state["puviani_nmf_surpass"] is None
    assert state["external_sota"] is None
    assert state["hardware_measured"] is None
    assert state["rank"] is None


def test_all_gates_and_targeted_mutations_pass():
    report = load(SEAL)
    assert report["gate_summary"] == {
        "passed": 20,
        "total": 20,
        "all_passed": True,
    }
    assert report["mutation_summary"] == {
        "detected": 20,
        "total": 20,
        "all_detected": True,
    }


def test_each_mutation_really_kills_its_target_gate():
    report = contract.build_report(root=ROOT, config_path=CONFIG)
    for _mutation_id, target, mutate in contract._mutations():
        changed = copy.deepcopy(report)
        mutate(changed)
        assert contract._evaluate(changed, ROOT)[target] is False


def test_config_rejects_demo_sized_samples():
    config = load(CONFIG)
    config["formal_matrix"][
        "samples_per_shared_state_action_backend"
    ] = 1
    with pytest.raises(ValueError, match="demo-sized"):
        contract._validate_config(config)


def test_config_rejects_formal_access_and_claim_boundary_drift():
    accessed = load(CONFIG)
    accessed["formal_result_accessed_before_freeze"] = True
    with pytest.raises(ValueError, match="accessed"):
        contract._validate_config(accessed)
    relabelled = load(CONFIG)
    relabelled["claim_boundary"]["typed_null"].remove("rank")
    with pytest.raises(ValueError, match="typed-null"):
        contract._validate_config(relabelled)


def test_json_has_no_nonfinite_values_and_analysis_is_reproducible():
    stored = load(SEAL)
    rebuilt = contract.build_report(root=ROOT, config_path=CONFIG)
    assert stored["analysis_sha256"] == rebuilt["analysis_sha256"]
    encoded = json.dumps(stored, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
