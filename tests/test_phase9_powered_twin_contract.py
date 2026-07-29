from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    CONFIG_PATH,
    EXPECTED_LABELS,
    build_cell_plan,
    cluster_root_id,
    heldout_seed,
    load_config,
    physical_seed,
    plan_payload,
    seed_registry_payload,
    validate_config,
)
from cnn_fpga.benchmark.phase9_powered_twin_blueprint_amendment import (
    PATTERN as FAULT_CUTOFF_PATTERN,
    build_amendment,
)
from cnn_fpga.benchmark.phase9_powered_twin_plan import (
    _assert_no_formal_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_live_config_parent_bindings_and_plan_fingerprint() -> None:
    config, binding = load_config(ROOT)
    plan = plan_payload(config)
    assert binding["path"] == CONFIG_PATH
    assert plan["cell_count"] == 518
    assert plan["row_count"] == 2_085_888
    assert plan["primary_density_count"] == 482_304
    assert (
        plan["canonical_plan_sha256"]
        == config["plan_contract"]["canonical_plan_sha256"]
    )


def test_exact_layer_cutoff_backend_and_row_accounting() -> None:
    cells = build_cell_plan(_config())
    assert Counter(cell.layer for cell in cells) == {
        "shared": 210,
        "logical": 252,
        "fault": 24,
        "probe": 32,
    }
    assert Counter((cell.layer, cell.cutoff) for cell in cells) == {
        ("shared", 36): 70,
        ("shared", 40): 70,
        ("shared", 44): 70,
        ("logical", 36): 84,
        ("logical", 40): 84,
        ("logical", 44): 84,
        ("fault", 36): 8,
        ("fault", 40): 8,
        ("fault", 44): 8,
        ("probe", 36): 32,
    }
    assert Counter(cell.backend for cell in cells) == {"A": 259, "B": 259}
    assert len({cell.chunk_id for cell in cells}) == 518
    assert [cell.plan_index for cell in cells] == list(range(518))
    assert sum(cell.expected_rows for cell in cells) == 2_085_888


def test_density_and_all_reset_scope_accounting() -> None:
    cells = build_cell_plan(_config())
    retained = {
        layer: sum(
            cell.sample_count
            for cell in cells
            if cell.layer == layer and cell.density_retention != "none"
        )
        for layer in ("shared", "logical", "probe", "fault")
    }
    assert retained == {
        "shared": 322_560,
        "logical": 0,
        "probe": 49_152,
        "fault": 110_592,
    }
    reset_cells = [
        cell for cell in cells if cell.reset_estimand_scope != "none"
    ]
    assert Counter(cell.layer for cell in reset_cells) == {
        "shared": 30,
        "logical": 36,
        "probe": 4,
        "fault": 6,
    }
    assert all(
        cell.action == "RESET"
        for cell in reset_cells
        if cell.layer != "fault"
    )
    assert all(cell.scenario == "compound" for cell in reset_cells if cell.layer == "fault")


def test_fault_denominator_is_state_major_6x768_without_modulo() -> None:
    config = _config()
    fault = next(
        cell
        for cell in build_cell_plan(config)
        if cell.layer == "fault" and cell.cutoff == 36 and cell.backend == "A"
    )
    roots = [cluster_root_id(config, fault, position) for position in range(4608)]
    states = [
        root.split("/state=", 1)[1].split("/", 1)[0]
        for root in roots
    ]
    assert Counter(states) == {label: 768 for label in EXPECTED_LABELS}
    assert states[:768] == ["0"] * 768
    assert states[768:1536] == ["1"] * 768
    assert states[-768:] == ["-i"] * 768
    assert roots[0].endswith("state=0/cluster=0000")
    assert roots[767].endswith("state=0/cluster=0767")
    assert roots[768].endswith("state=1/cluster=0000")


def test_seed_registry_enumerates_actual_addresses_and_registered_pairing() -> None:
    config = _config()
    cells = build_cell_plan(config)
    registry = seed_registry_payload(config)
    assert registry["actual_unique_physical_addresses"] == 322_560
    assert registry["actual_unique_heldout_addresses"] == 364_032
    by_identity: dict[tuple[str, str, int], list[object]] = defaultdict(list)
    for cell in cells:
        by_identity[(cell.layer, cell.cell_base, cell.backend)].append(cell)
    # Cutoff is intentionally omitted for within-backend CRN.
    shared = sorted(
        by_identity[("shared", "shared|vacuum_g|IDLE", "A")],
        key=lambda cell: cell.cutoff,
    )
    assert len(shared) == 3
    assert len({physical_seed(config, cell, 17) for cell in shared}) == 1
    assert len({heldout_seed(config, cell, 17, 0) for cell in shared}) == 1
    # Backend-native physical streams are disjoint; heldout IQ is common.
    a = shared[0]
    b = next(
        cell
        for cell in cells
        if cell.layer == a.layer
        and cell.cell_base == a.cell_base
        and cell.cutoff == a.cutoff
        and cell.backend == "B"
    )
    assert physical_seed(config, a, 17) != physical_seed(config, b, 17)
    assert heldout_seed(config, a, 17, 0) == heldout_seed(config, b, 17, 0)
    # Independent pair groups never alias merely because position matches.
    other = next(
        cell
        for cell in cells
        if cell.layer == "shared"
        and cell.cell_base == "shared|one_g|IDLE"
        and cell.cutoff == 36
        and cell.backend == "A"
    )
    assert physical_seed(config, a, 17) != physical_seed(config, other, 17)
    assert heldout_seed(config, a, 17, 0) != heldout_seed(config, other, 17, 0)


@pytest.mark.parametrize(
    ("locator", "replacement", "message"),
    [
        (("formal_matrix", "round_clusters_per_cell"), 1535, "formal_matrix"),
        (("formal_matrix", "fault_state_ordering"), "position_mod_6", "state ordering"),
        (
            ("formal_matrix", "fault_scenario_parameters", "step", "change_round"),
            5,
            "fault scenario parameter",
        ),
        (
            ("formal_matrix", "fault_intervention_witness", "dtype"),
            "<f4",
            "intervention witness",
        ),
        (
            (
                "resource_contract",
                "profile_plan",
                "retained_density_physicality_full_482304",
                "full_retained_count",
            ),
            482303,
            "resource preflight execution plan",
        ),
        (
            (
                "resource_contract",
                "profile_plan",
                "retained_density_physicality_full_482304",
                "block_size",
            ),
            16,
            "resource preflight execution plan",
        ),
        (
            (
                "resource_contract",
                "profile_plan",
                "retained_density_physicality_full_482304",
                "fixture_matrix_count",
            ),
            1,
            "resource preflight execution plan",
        ),
        (
            (
                "resource_contract",
                "profile_plan",
                "retained_density_physicality_full_482304",
                "sampled",
            ),
            True,
            "resource preflight execution plan",
        ),
        (("statistics_contract", "multiplier_replicates"), 198, "statistics"),
        (("statistics_contract", "influence_source"), "synthetic", "statistics"),
        (("reset_contract", "all_reset_scopes_required"), False, "RESET"),
        (("transaction_contract", "single_monolithic_zip_forbidden"), False, "transaction"),
        (("seed_registry", "physical", "start"), 100_010_000_000, "overlap"),
        (("claim_boundary", "external_sota"), True, "literal null"),
    ],
)
def test_contract_mutations_fail_closed(
    locator: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    config = copy.deepcopy(_config())
    target = config
    for key in locator[:-1]:
        target = target[key]
    target[locator[-1]] = replacement
    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_config(config)


def test_blueprint_or_parent_byte_mutation_is_detected(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    config["blueprint_binding"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="blueprint"):
        validate_config(config, root=ROOT)


def test_effective_blueprint_changes_only_the_16_fault_cutoff_denominators() -> None:
    selected = json.loads(
        (
            ROOT / "docs/t_risk_20260728_06_selected_gate_blueprint.json"
        ).read_text(encoding="utf-8")
    )
    rebuilt = build_amendment(ROOT)
    published = json.loads(
        (
            ROOT / "docs/t_risk_20260728_04_effective_gate_blueprint.json"
        ).read_text(encoding="utf-8")
    )
    assert rebuilt == published
    assert rebuilt["changed_gate_count"] == 16
    changes = {item["gate_id"]: item for item in rebuilt["changes"]}
    assert len(changes) == 16
    for original, effective in zip(
        selected["gates"], rebuilt["gates"], strict=True
    ):
        gate_id = original["gate_id"]
        differing = {
            key
            for key in set(original) | set(effective)
            if original.get(key) != effective.get(key)
        }
        if FAULT_CUTOFF_PATTERN.fullmatch(gate_id):
            assert differing == {"cluster_count", "stage", "cluster_scope"}
            assert effective["cluster_count"] == 4608
            assert effective["stage"] == "trajectory"
            assert effective["cluster_scope"].endswith("/all_states")
            assert gate_id in changes
            assert effective["margin"] == original["margin"]
            assert effective["gate_id"] == original["gate_id"]
        else:
            assert differing == set()
    assert rebuilt["scientific_margin_changed"] is False
    assert rebuilt["postselection_used"] is False
    assert rebuilt["cross_state_averaging_used"] is False
    assert rebuilt["qualified_claim"] is None


def test_plan_hash_is_key_order_independent_but_semantics_sensitive() -> None:
    config = _config()
    reordered = json.loads(
        json.dumps(config, sort_keys=True, ensure_ascii=False)
    )
    assert (
        plan_payload(config)["canonical_plan_sha256"]
        == plan_payload(reordered)["canonical_plan_sha256"]
    )
    mutated = copy.deepcopy(config)
    mutated["formal_matrix"]["probe_actions"][8]["action"] = "HOLD"
    with pytest.raises(ValueError, match="probe action"):
        plan_payload(mutated)


def test_out_of_range_seed_and_cluster_addresses_are_rejected() -> None:
    config = _config()
    cell = build_cell_plan(config)[0]
    for invalid in (-1, cell.sample_count):
        with pytest.raises(ValueError):
            physical_seed(config, cell, invalid)
        with pytest.raises(ValueError):
            cluster_root_id(config, cell, invalid)
    with pytest.raises(ValueError):
        heldout_seed(config, cell, 0, cell.horizon)


@pytest.mark.parametrize(
    ("path_key", "message"),
    [
        ("staging_directory", "staging payload"),
        ("independent_verification", "finalize artifact"),
    ],
)
def test_contract_materialization_rejects_late_formal_artifacts(
    tmp_path: Path,
    path_key: str,
    message: str,
) -> None:
    config = copy.deepcopy(_config())
    for key in (
        "object_store",
        "staging_directory",
        "receipt_directory",
        "inventory",
        "execution_manifest",
        "independent_verification",
    ):
        config["artifact_paths"][key] = f"isolated/{key}"
    target = tmp_path / config["artifact_paths"][path_key]
    if path_key == "staging_directory":
        target = target / "payload.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"late-formal-evidence")
    with pytest.raises(RuntimeError, match=message):
        _assert_no_formal_artifacts(tmp_path, config)
