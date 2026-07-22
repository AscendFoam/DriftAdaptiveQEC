from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from cnn_fpga.benchmark import puviani_paper_constrained_artifacts as subject
from physics.fock_logical_channel import STATE_LABELS


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs/phase9/t9_1_3_puviani_paper_constrained.json"


def _pilot_config() -> dict[str, object]:
    payload = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    training = payload["training"]
    qualification = payload["six_state_qualification"]
    artifact = payload["artifact_contract"]
    training.update(
        {
            "epochs": 3,
            "epochs_by_strategy": {"mf": 3, "nmf": 3},
            "full_cycles": 2,
            "cutoff": 4,
            "confirmation_cutoff": 4,
            "device": "cpu",
            "paired_root_seeds": [981_001],
            "validation_seeds": [982_003],
            "validation_interval": 1,
            "mf_batch_size": 1,
            "nmf_batch_size": 1,
            "validation_batch_size": 1,
            "bootstrap_repetitions": 10,
            "seed_derivation": {
                "training_rollout_seed": "paired_root_seed*1000003+epoch",
                "paired_across_mf_and_nmf": True,
                "root_seed_count": 1,
                "strategy_count": 2,
                "independently_trained_agent_count": 2,
            },
        }
    )
    qualification.update(
        {
            "states": list(STATE_LABELS),
            "seeds": [983_007],
            "confirmation_seeds": [984_009],
            "batch_size": 1,
            "confirmation_batch_size": 1,
            "full_cycles": 2,
            "primary_cutoff": 4,
            "confirmation_cutoff": 4,
        }
    )
    artifact.update(
        {
            "agent_manifest_rows_expected": 2,
            "selected_agent_rows_expected": 2,
        }
    )
    return payload


@pytest.fixture(scope="module")
def lineage_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    pytest.importorskip("torch")
    config = _pilot_config()
    output = tmp_path_factory.mktemp("puviani_lineage_template") / "agents"
    result = subject.train_population(
        config,
        output_dir=output,
        family="mf",
        production=False,
    )
    assert result["newly_trained_agents"] == 1
    return config, output


def _copy_bundle(
    template: tuple[dict[str, object], Path], tmp_path: Path
) -> tuple[dict[str, object], Path, int]:
    config, source = template
    destination = tmp_path / "agents"
    shutil.copytree(source, destination)
    seed = int(config["training"]["paired_root_seeds"][0])
    weight_path, terminal_path, record_path = subject._agent_paths(
        destination, "mf", seed
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["weight_path"] = subject._portable_artifact_path(weight_path)
    record["terminal_path"] = subject._portable_artifact_path(terminal_path)
    subject._atomic_json(record, record_path)
    return config, destination, seed


def _load(config: dict[str, object], output: Path, seed: int):
    return subject._load_agent_record(
        output_dir=output,
        strategy="mf",
        seed=seed,
        directional=subject._strategy_config(config, "mf"),
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )


def _record_and_terminal(output: Path, seed: int):
    torch = pytest.importorskip("torch")
    _, terminal_path, record_path = subject._agent_paths(output, "mf", seed)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    return record, terminal, terminal_path, record_path


def _write_mutated_terminal(
    record: dict[str, object],
    terminal: dict[str, object],
    terminal_path: Path,
    record_path: Path,
) -> None:
    subject._atomic_torch(terminal, terminal_path)
    record["terminal_file_sha256"] = subject._file_sha256(terminal_path)
    record["terminal_bytes"] = terminal_path.stat().st_size
    subject._atomic_json(record, record_path)


def test_tiny_pilot_closes_actual_epoch_endpoints_and_chain(
    lineage_template: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    config, output, seed = _copy_bundle(lineage_template, tmp_path)
    loaded = _load(config, output, seed)
    assert loaded is not None
    _, record = loaded
    training = record["training_record"]
    witness = record["live_training_lineage_validation"]
    _, terminal, _, _ = _record_and_terminal(output, seed)
    candidates = {row["epoch"]: row for row in terminal["validation_candidate_states"]}
    curve = training["training_curve"]
    assert record["training_lineage_reload_witness"] is True
    assert witness["all_epochs_closed"] is True
    assert witness["endpoints_bound"] is True
    assert subject.state_dict_sha256(candidates[0]["model_state"]) == training[
        "initial_state_sha256"
    ]
    assert subject.state_dict_sha256(
        candidates[3]["model_state"]
    ) == subject.state_dict_sha256(terminal["terminal_model_state"])
    assert curve[0]["previous_lineage_sha256"] == training[
        "training_lineage_genesis_sha256"
    ]
    assert curve[-1]["lineage_sha256"] == training[
        "training_lineage_terminal_sha256"
    ]


def test_initial_hash_replacement_is_rejected(
    lineage_template: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    config, output, seed = _copy_bundle(lineage_template, tmp_path)
    record, _, _, record_path = _record_and_terminal(output, seed)
    record["training_record"]["initial_state_sha256"] = "1" * 64
    subject._atomic_json(record, record_path)
    with pytest.raises(subject.CorruptAgentArtifactError, match="endpoint binding"):
        _load(config, output, seed)


def test_terminal_endpoint_swap_is_rejected_even_when_file_hash_is_resealed(
    lineage_template: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    config, output, seed = _copy_bundle(lineage_template, tmp_path)
    record, terminal, terminal_path, record_path = _record_and_terminal(output, seed)
    initial = deepcopy(terminal["validation_candidate_states"][0]["model_state"])
    swapped_hash = subject.state_dict_sha256(initial)
    terminal["terminal_model_state"] = initial
    terminal["terminal_model_sha256"] = swapped_hash
    record["training_record"]["terminal_state_sha256"] = swapped_hash
    _write_mutated_terminal(record, terminal, terminal_path, record_path)
    with pytest.raises(subject.CorruptAgentArtifactError, match="endpoint binding"):
        _load(config, output, seed)


@pytest.mark.parametrize("mutation", ("finite_metric", "epoch", "chain"))
def test_finite_metric_epoch_and_hash_chain_tampering_are_rejected(
    lineage_template: tuple[dict[str, object], Path],
    tmp_path: Path,
    mutation: str,
) -> None:
    config, output, seed = _copy_bundle(lineage_template, tmp_path)
    record, _, _, record_path = _record_and_terminal(output, seed)
    curve = record["training_record"]["training_curve"]
    if mutation == "finite_metric":
        curve[1]["mean_reward"] = float(curve[1]["mean_reward"]) + 1.0e-3
    elif mutation == "epoch":
        curve[1]["epoch"] = 99
    else:
        curve[1]["lineage_sha256"] = "a" * 64
    subject._atomic_json(record, record_path)
    with pytest.raises(subject.CorruptAgentArtifactError):
        _load(config, output, seed)


def test_validation_candidate_state_must_equal_corresponding_epoch_post_hash(
    lineage_template: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    config, output, seed = _copy_bundle(lineage_template, tmp_path)
    record, terminal, terminal_path, record_path = _record_and_terminal(output, seed)
    best_epoch = int(record["training_record"]["best_validation_epoch"])
    interior_epoch = next(epoch for epoch in (1, 2) if epoch != best_epoch)
    candidate = terminal["validation_candidate_states"][interior_epoch]
    candidate["model_state"] = deepcopy(
        terminal["validation_candidate_states"][0]["model_state"]
    )
    forged_hash = subject.state_dict_sha256(candidate["model_state"])
    candidate["state_sha256"] = forged_hash
    record["training_record"]["validation_history"][interior_epoch][
        "candidate_state_sha256"
    ] = forged_hash
    _write_mutated_terminal(record, terminal, terminal_path, record_path)
    with pytest.raises(
        subject.CorruptAgentArtifactError,
        match="validation candidate is not the epoch post-state",
    ):
        _load(config, output, seed)


@pytest.fixture(scope="module")
def sentinel_population(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    dict[str, object],
    dict[str, list[tuple[object, dict[str, object]]]],
    Path,
]:
    pytest.importorskip("torch")
    config = _pilot_config()
    root = tmp_path_factory.mktemp("puviani_sentinel_population")
    output = root / "agents"
    result = subject.train_population(
        config,
        output_dir=output,
        family="all",
        production=False,
    )
    assert result["newly_trained_agents"] == 2
    seed = int(config["training"]["paired_root_seeds"][0])
    agents: dict[str, list[tuple[object, dict[str, object]]]] = {
        "mf": [],
        "nmf": [],
    }
    for strategy in ("mf", "nmf"):
        loaded = subject._load_agent_record(
            output_dir=output,
            strategy=strategy,
            seed=seed,
            directional=subject._strategy_config(config, strategy),
            config=config,
            implementation_hash=subject.implementation_sha256(),
        )
        assert loaded is not None
        agents[strategy].append(loaded)
    return config, agents, root


def test_two_family_sentinel_full_retrain_matches_all_lineage_evidence(
    sentinel_population: tuple[
        dict[str, object],
        dict[str, list[tuple[object, dict[str, object]]]],
        Path,
    ],
) -> None:
    config, agents, root = sentinel_population
    audit = subject._sentinel_full_retrain_audit(
        config,
        agents,
        scratch_root=root / "sentinel_scratch",
    )
    assert audit["sentinel_agent_count"] == 2
    assert audit["population_agent_count"] == 2
    assert audit["claim_scope"] == "two_of_2_pilot_agents"
    assert audit["all_sentinels_exactly_reproduced"] is True
    assert audit["temporary_artifacts_deleted"] is True
    assert subject._sentinel_retrain_audit_valid(
        audit,
        expected_epochs=3,
        expected_population_agents=2,
    )


def test_sentinel_retrain_rejects_a_coordinated_finite_training_ledger_tamper(
    sentinel_population: tuple[
        dict[str, object],
        dict[str, list[tuple[object, dict[str, object]]]],
        Path,
    ],
) -> None:
    config, agents, root = sentinel_population
    tampered = {
        strategy: [(model, deepcopy(record)) for model, record in family]
        for strategy, family in agents.items()
    }
    tampered["mf"][0][1]["training_record"]["training_curve"][1][
        "mean_reward"
    ] += 1.0e-6
    with pytest.raises(
        subject.CorruptAgentArtifactError,
        match="not reproducible|sentinel retrain evidence mismatch",
    ):
        subject._sentinel_full_retrain_audit(
            config,
            tampered,
            scratch_root=root / "tampered_sentinel_scratch",
        )


def test_sentinel_compact_audit_mutation_revokes_gate(
    sentinel_population: tuple[
        dict[str, object],
        dict[str, list[tuple[object, dict[str, object]]]],
        Path,
    ],
) -> None:
    config, agents, root = sentinel_population
    audit = subject._sentinel_full_retrain_audit(
        config,
        agents,
        scratch_root=root / "sentinel_audit_mutation_scratch",
    )
    forged = deepcopy(audit)
    forged["families"][0]["terminal_model_sha256"] = "f" * 64
    assert not subject._sentinel_retrain_audit_valid(
        forged,
        expected_epochs=3,
        expected_population_agents=2,
    )
