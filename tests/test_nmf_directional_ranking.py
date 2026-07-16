from __future__ import annotations

import json
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from physics.differentiable_sbs_trajectory import (  # noqa: E402
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
)
from physics.nmf_directional_ranking import (  # noqa: E402
    ANALYSIS_CONTRACT_ID,
    DirectionalRankingConfig,
    NMFLatestOnlyView,
    PAPER_MF_ARCHITECTURE,
    PAPER_RNN_ARCHITECTURE,
    POLICY_INITIALIZATION_ID,
    TRAINING_PROTOCOL_ID,
    PaperScaleNMFPolicy,
    _area_equivalent_lifetime,
    _effective_lifetime,
    build_policy,
    evaluate_policy,
    implementation_sha256,
    run_directional_ranking,
    state_dict_sha256,
    train_agent,
    validate_production_design,
)
from physics.plot_nmf_directional_ranking import (  # noqa: E402
    load_and_audit_artifact,
    render,
)

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ARTIFACT = ROOT / "docs/t2_3_7_nmf_directional_ranking.json"
PRODUCTION_SOURCE_DATA = ROOT / "docs/t2_3_7_nmf_directional_ranking.csv"


def tiny_config(**overrides: object) -> DirectionalRankingConfig:
    values: dict[str, object] = {
        "cutoff": 6,
        "confirmation_cutoff": 6,
        "full_cycles": 2,
        "train_epochs": 2,
        "train_batch_size": 2,
        "validation_batch_size": 2,
        "test_batch_size": 2,
        "confirmation_batch_size": 2,
        "validation_interval": 1,
        "training_seeds": (11,),
        "validation_seeds": (101,),
        "test_seeds": (211,),
        "confirmation_seeds": (307,),
        "bootstrap_repetitions": 100,
        "device": "cpu",
    }
    values.update(overrides)
    return DirectionalRankingConfig(**values)


def test_default_contract_is_paper_horizon_and_disjoint() -> None:
    config = DirectionalRankingConfig(device="cpu")
    assert config.full_cycles == 10
    assert config.analysis_contract_id == ANALYSIS_CONTRACT_ID
    assert config.policy_initialization_id == POLICY_INITIALIZATION_ID
    assert TRAINING_PROTOCOL_ID.endswith("TRAIN-ONLY-EMA-BASELINE-V1")
    assert len(implementation_sha256()) == 64
    assert not set(config.training_seeds) & set(config.test_seeds)
    assert config.train_trajectories_per_family == 12_800
    assert config.test_trajectories_per_agent == 512


@pytest.mark.parametrize(
    "updates",
    [
        {"training_seeds": (1,), "validation_seeds": (1,)},
        {"full_cycles": 11},
        {"score_baseline_decay": 1.0},
        {"analysis_contract_id": "stale"},
        {"policy_initialization_id": "stale"},
    ],
)
def test_invalid_contracts_fail_closed(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        tiny_config(**updates)


def test_production_design_rejects_pilot() -> None:
    with pytest.raises(ValueError, match="production ranking"):
        validate_production_design(tiny_config())


def test_paper_scale_parameter_counts_and_labels() -> None:
    config = tiny_config()
    mf = build_policy("mf", config, 11)
    nmf = build_policy("nmf", config, 11)
    assert mf.parameter_count == 70_159
    assert nmf.parameter_count == 72_853
    assert PAPER_MF_ARCHITECTURE.endswith("LATEST-OUTCOME")
    assert PAPER_RNN_ARCHITECTURE == "GRU10-DENSE256-DENSE256-OUT15"


def test_mf_is_invariant_to_earlier_history_with_same_latest_outcome() -> None:
    model = build_policy("mf", tiny_config(), 11)
    history = torch.tensor([[0, 0, 1], [1, 0, 1]], dtype=torch.int64)
    output = model(history, 3)
    assert torch.allclose(output[0], output[1], atol=0.0, rtol=0.0)


def test_nmf_can_distinguish_histories_with_the_same_latest_outcome() -> None:
    model = build_policy("nmf", tiny_config(), 11)
    history = torch.tensor([[0, 0, 1], [1, 0, 1]], dtype=torch.int64)
    output = model(history, 3)
    assert torch.max(torch.abs(output[0] - output[1])).detach().item() > 1.0e-8


def test_cached_recurrent_rollout_matches_full_history_replay_exactly() -> None:
    model = build_policy("nmf", tiny_config(), 11)
    history = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.int64)
    model.reset_rollout(batch_size=2, device=torch.device("cpu"), dtype=torch.float64)
    for half_index in range(5):
        prefix = history[:, :half_index]
        cached = model.step_rollout(prefix, half_index)
        replay = model(prefix, half_index)
        assert torch.allclose(cached, replay, atol=0.0, rtol=0.0)


def test_latest_only_view_is_the_hidden_reset_ablation() -> None:
    model = build_policy("nmf", tiny_config(), 11)
    view = NMFLatestOnlyView(model)
    history = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64)
    assert torch.allclose(
        view(history, 3),
        model.forward_with_mode(history, 3, latest_only=True),
        atol=0.0,
        rtol=0.0,
    )


def test_cycle_metrics_include_time_zero_and_every_full_cycle() -> None:
    simulator = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(cutoff=6, full_cycles=2, batch_size=3)
    )
    result = simulator.run(seed=17, record_cycle_metrics=True)
    assert result.cycle_fidelities.shape == (3, 3)
    assert result.cycle_code_survival.shape == (3, 3)
    assert result.cycle_logical_z_signal.shape == (3, 3)
    assert result.cycle_conditional_logical_z.shape == (3, 3)
    assert torch.allclose(
        result.cycle_fidelities[:, 0],
        torch.ones(3, dtype=result.cycle_fidelities.dtype),
        atol=1.0e-12,
    )
    assert torch.all((result.cycle_code_survival >= 0.0) & (result.cycle_code_survival <= 1.0))


def test_cycle_metrics_are_opt_in_to_preserve_training_memory() -> None:
    simulator = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(cutoff=6, full_cycles=1, batch_size=2)
    )
    result = simulator.run(seed=17)
    assert result.cycle_fidelities is None
    assert result.cycle_logical_z_signal is None


@pytest.mark.parametrize("lifetime", [0.5, 2.0, 10.0, 100.0])
def test_area_equivalent_lifetime_inverts_analytic_area(lifetime: float) -> None:
    horizon = 10.0
    area = (lifetime / horizon) * (1.0 - np.exp(-horizon / lifetime))
    recovered = _area_equivalent_lifetime(area, horizon)
    assert recovered == pytest.approx(lifetime, rel=1.0e-10, abs=1.0e-10)


def test_effective_lifetime_reports_area_method_and_fit_only_as_diagnostic() -> None:
    curve = np.asarray([1.0, 0.65, 0.72, 0.55, 0.50])
    result = _effective_lifetime(curve)
    assert result["effective_lifetime_method"] == "finite_horizon_area_equivalent_exponential"
    assert result["exponential_fit_is_diagnostic_only"] is True
    assert result["effective_lifetime_cycles"] > 0.0
    assert np.isfinite(result["log_linear_fit_r_squared_diagnostic"])


def test_better_curve_has_larger_area_equivalent_lifetime() -> None:
    weak = _effective_lifetime([1.0, 0.7, 0.5, 0.3])
    strong = _effective_lifetime([1.0, 0.8, 0.7, 0.6])
    assert strong["normalized_auc"] > weak["normalized_auc"]
    assert strong["effective_lifetime_cycles"] > weak["effective_lifetime_cycles"]


def test_evaluate_policy_is_seed_reproducible_and_reports_same_physical_time() -> None:
    config = tiny_config()
    first = evaluate_policy(
        "standard", None, config, cutoff=6, batch_size=2, seeds=(211,)
    )
    second = evaluate_policy(
        "standard", None, config, cutoff=6, batch_size=2, seeds=(211,)
    )
    assert first["per_seed"] == second["per_seed"]
    assert first["simulated_physical_time_us"] == 20.0
    assert first["total_trajectories"] == 2


def test_evaluate_policy_requires_correct_model_kind() -> None:
    config = tiny_config()
    with pytest.raises(ValueError, match="must not provide"):
        evaluate_policy("standard", build_policy("mf", config, 11), config, cutoff=6, batch_size=2, seeds=(211,))
    with pytest.raises(TypeError, match="PaperScaleNMFPolicy"):
        evaluate_policy(
            "nmf_latest_only",
            build_policy("mf", config, 11),
            config,
            cutoff=6,
            batch_size=2,
            seeds=(211,),
        )


def test_train_agent_uses_nonzero_feedback_grape_gradients_and_validation_selection() -> None:
    model, record = train_agent("nmf", 11, tiny_config())
    assert isinstance(model, PaperScaleNMFPolicy)
    assert record["epochs_executed"] == 2
    assert record["best_validation_epoch"] in {0, 1, 2}
    assert record["baseline_initialization"] == "train-only no-gradient warm-up reward mean"
    assert record["baseline_warmup_seed"] not in tiny_config().validation_seeds
    assert set(record["training_trajectory_seeds"]).isdisjoint(
        tiny_config().validation_seeds
        + tiny_config().test_seeds
        + tiny_config().confirmation_seeds
    )
    best = max(record["validation_history"], key=lambda item: item["selection_score"])
    assert record["best_validation_epoch"] == best["epoch"]
    assert record["best_validation_score"] == best["selection_score"]
    assert all(item["gradient_norm_before_clip"] > 0.0 for item in record["training_curve"])
    assert state_dict_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()}) == record[
        "checkpoint_sha256"
    ]


def test_pilot_artifact_persists_all_models_and_resumes_by_contract(tmp_path: Path) -> None:
    config = tiny_config(train_epochs=1, validation_interval=1)
    artifact_path = tmp_path / "ranking.json"
    checkpoint_path = tmp_path / "ranking.pt"
    first = run_directional_ranking(
        config,
        artifact_path=artifact_path,
        checkpoint_path=checkpoint_path,
        production=False,
    )
    assert first["execution"]["newly_trained_agents"] == 2
    assert first["checkpoint"]["contains_all_training_seed_models"]
    assert first["checkpoint"]["all_model_hashes_match"]
    assert first["checkpoint"]["schema_version"] == 3
    assert first["gates"]["checkpoint_selection_uses_validation_only"]
    assert first["gates"]["checkpoint_contains_all_models_with_matching_hashes"]
    assert first["gates"]["nonexponential_fit_is_diagnostic_only"]
    assert first["training_protocol_id"] == TRAINING_PROTOCOL_ID
    assert first["implementation_sha256"] == implementation_sha256()
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["task_id"] == "T2.3.7"
    second = run_directional_ranking(
        config,
        artifact_path=artifact_path,
        checkpoint_path=checkpoint_path,
        production=False,
    )
    assert second["execution"]["resumed_agents"] == 2
    assert second["execution"]["newly_trained_agents"] == 0


def test_resume_rejects_changed_contract(tmp_path: Path) -> None:
    config = tiny_config(train_epochs=1, validation_interval=1)
    artifact_path = tmp_path / "ranking.json"
    checkpoint_path = tmp_path / "ranking.pt"
    run_directional_ranking(
        config,
        artifact_path=artifact_path,
        checkpoint_path=checkpoint_path,
        production=False,
    )
    changed = replace(config, learning_rate=2.0e-4)
    with pytest.raises(ValueError, match="contract hash"):
        run_directional_ranking(
            changed,
            artifact_path=artifact_path,
            checkpoint_path=checkpoint_path,
            production=False,
        )


def test_resume_rejects_source_or_training_protocol_drift(tmp_path: Path) -> None:
    config = tiny_config(train_epochs=1, validation_interval=1)
    artifact_path = tmp_path / "ranking.json"
    checkpoint_path = tmp_path / "ranking.pt"
    run_directional_ranking(
        config,
        artifact_path=artifact_path,
        checkpoint_path=checkpoint_path,
        production=False,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["checkpoint_contract_hash"] = "0" * 64
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="config/source/training protocol"):
        run_directional_ranking(
            config,
            artifact_path=artifact_path,
            checkpoint_path=checkpoint_path,
            production=False,
        )


def test_production_artifact_passes_independent_non_demo_audit() -> None:
    payload = load_and_audit_artifact(PRODUCTION_ARTIFACT)
    assert payload["status"] == "PASS"
    assert payload["config"]["full_cycles"] == 10
    assert payload["config"]["test_batch_size"] * len(payload["config"]["test_seeds"]) == 512
    assert len(payload["training_records"]["mf"]) == 5
    assert len(payload["training_records"]["nmf"]) == 5
    assert all(payload["gates"][name] for name in payload["required_directional_gates"])


def test_production_ranking_keeps_all_agents_and_cutoff_ablation_counterevidence() -> None:
    payload = json.loads(PRODUCTION_ARTIFACT.read_text(encoding="utf-8"))
    primary = payload["summary"]["primary"]
    confirmation = payload["summary"]["confirmation"]
    mf = np.asarray(primary["mf"]["logical_z_effective_lifetime_cycles"]["values"])
    nmf = np.asarray(primary["nmf"]["logical_z_effective_lifetime_cycles"]["values"])
    reset = np.asarray(
        primary["nmf_latest_only"]["logical_z_effective_lifetime_cycles"]["values"]
    )
    assert mf.shape == nmf.shape == reset.shape == (5,)
    assert np.all(nmf > mf)
    assert np.all(nmf > reset)
    assert payload["paired_bootstrap"]["nmf_minus_mf_logical_z_lifetime"]["ci95_low"] > 0.0
    assert (
        confirmation["nmf_latest_only"]["logical_z_effective_lifetime_cycles"]["median"]
        > confirmation["nmf"]["logical_z_effective_lifetime_cycles"]["median"]
    )


def test_source_data_and_publication_bundle_are_traceable() -> None:
    with PRODUCTION_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 8_000
    assert sum(row["record_type"] == "curve" for row in rows) == 6_336
    assert {
        row["strategy"]
        for row in rows
        if row["record_type"] == "curve"
    } == {"standard", "mf", "nmf", "nmf_latest_only"}
    figure_stem = ROOT / "docs/figures/t2_3_7_nmf_directional_ranking"
    for suffix in ("svg", "pdf", "tiff", "png"):
        assert figure_stem.with_suffix(f".{suffix}").stat().st_size > 10_000
    svg = figure_stem.with_suffix(".svg").read_text(encoding="utf-8")
    assert svg.count("<text") >= 50


def test_plot_audit_rejects_stale_contract(tmp_path: Path) -> None:
    payload = json.loads(PRODUCTION_ARTIFACT.read_text(encoding="utf-8"))
    payload["analysis_contract_id"] = "stale"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="analysis contract is stale"):
        load_and_audit_artifact(path, verify_checkpoint=False)


def test_plot_render_recreates_all_formats_and_source_data(tmp_path: Path) -> None:
    outputs = render(
        PRODUCTION_ARTIFACT,
        tmp_path / "ranking",
        tmp_path / "source.csv",
    )
    assert {path.suffix for path in outputs} == {".svg", ".pdf", ".tiff", ".png"}
    assert all(path.stat().st_size > 10_000 for path in outputs)
    assert (tmp_path / "source.csv").stat().st_size > 100_000
