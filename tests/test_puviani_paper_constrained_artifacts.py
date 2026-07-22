from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import time

import numpy as np
import pytest

from cnn_fpga.benchmark import puviani_paper_constrained_artifacts as subject
from physics.fock_logical_channel import STATE_LABELS


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs/phase9/t9_1_3_puviani_paper_constrained.json"
PRODUCTION_REPORT = ROOT / "docs/t9_1_3_puviani_paper_constrained.json"


def _load_production_config() -> dict[str, object]:
    payload = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _pilot_config() -> dict[str, object]:
    """Return a minimal executable config without weakening split isolation."""

    payload = deepcopy(_load_production_config())
    training = payload["training"]
    qualification = payload["six_state_qualification"]
    artifact = payload["artifact_contract"]
    assert isinstance(training, dict)
    assert isinstance(qualification, dict)
    assert isinstance(artifact, dict)

    training.update(
        {
            "epochs": 1,
            "epochs_by_strategy": {"mf": 1, "nmf": 1},
            "full_cycles": 2,
            "cutoff": 4,
            "confirmation_cutoff": 4,
            "device": "cpu",
            "paired_root_seeds": [991_001],
            "validation_seeds": [992_003],
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
            "seeds": [993_007],
            "confirmation_seeds": [994_009],
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


def _reseal(payload: dict[str, object]) -> dict[str, object]:
    payload.pop("analysis_sha256", None)
    payload["analysis_sha256"] = subject._canonical_sha256(payload)
    return payload


def _production_deterministic_signature(
    value: dict[str, object],
) -> dict[str, object]:
    signature = deepcopy(value)
    signature.update(
        {
            "deterministic_algorithms_enabled": True,
            "deterministic_debug_mode": 2,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
    )
    signature["environment_controls"].update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            "PYTHONHASHSEED": "0",
        }
    )
    return signature


def _minimal_valid_report() -> dict[str, object]:
    """Build a production-shaped in-memory seal without faking live files."""

    config = _load_production_config()
    root = Path(subject.__file__).resolve().parents[2]
    parent = subject._verify_parent_protocol(config)
    sources = subject._source_registry(config)
    runtime = subject._runtime_profile()
    runtime["signature"] = _production_deterministic_signature(
        runtime["signature"]
    )
    runtime["signature_sha256"] = subject._canonical_sha256(runtime["signature"])
    runtime_signature = deepcopy(runtime["signature"])
    runtime_hash = str(runtime["signature_sha256"])
    training_runtime_contract = {
        "schema_version": "t9.1.3-training-runtime-contract-v1",
        "signature": runtime_signature,
        "signature_sha256": runtime_hash,
        "agent_signature_count": 40,
        "distinct_agent_signature_sha256": [runtime_hash],
        "all_agents_identical": True,
        "finalizer_signature_sha256": runtime_hash,
        "equals_finalizer_runtime_signature": True,
    }
    deviation_path = root / str(config["deviation_ledger_path"])
    deviations = json.loads(deviation_path.read_text(encoding="utf-8"))["deviations"]
    tomography = {
        "ptm": np.eye(4, dtype=np.float64).tolist(),
        "choi_real": np.eye(4, dtype=np.float64).tolist(),
        "choi_imag": np.zeros((4, 4), dtype=np.float64).tolist(),
        "tni_effect_eigenvalues": [1.0, 1.0],
        "pair_sum_linearity_residual": 0.0,
        "minimum_choi_eigenvalue": 0.0,
        "maximum_output_hermiticity_error": 0.0,
        "minimum_output_eigenvalue": 0.0,
        "minimum_survival": 1.0,
        "maximum_survival": 1.0,
        "mean_leakage": 0.0,
        "survival_spread": 0.0,
    }
    diagnostic = {
        "selected_agent_seed": None,
        "trajectory_count": 0,
        "empirical_linear_inversion_subchannel_diagnostic": tomography,
        "monte_carlo_pair_sum_not_deterministic_tolerance": True,
        "empirical_cptni_passed_is_not_a_hard_gate": True,
        "state_definition_id": config["six_state_qualification"][
            "state_definition_id"
        ],
        "logical_operator_definition_id": config["six_state_qualification"][
            "logical_operator_definition_id"
        ],
        "monte_carlo_design": config["six_state_qualification"][
            "monte_carlo_design"
        ],
        "raw_codeword_overlap_real": 0.0,
        "raw_codeword_overlap_imag": 0.0,
        "raw_logical_gram_condition_number": 1.0,
        "diagnostic_scope": (
            "unconstrained linear inversion of paired-common-random-number six-state "
            "means clustered by stratum/evaluation seed; retain CP/TNI diagnostics "
            "but do not treat finite-sample violations as physical-channel proof"
        ),
        "maximum_joint_trace_error": 0.0,
        "maximum_joint_hermiticity_error": 0.0,
        "maximum_projected_hermiticity_error_before_stabilization": 0.0,
        "minimum_final_joint_eigenvalue": 0.0,
        "paper_lifetime_value": None,
        "paper_lifetime_state": subject.PAPER_NUMERIC_STATE,
    }
    reconstruction = {
        stratum: {
            method: deepcopy(diagnostic)
            for method in ("standard", "mf", "nmf")
        }
        for stratum in ("primary_cutoff12", "selected_confirmation_cutoff16")
    }
    roots = config["training"]["paired_root_seeds"]
    sentinel_families = [
        {
            "strategy": strategy,
            "root_seed": int(roots[0]),
            "epochs_retrained": 1_000,
            "validation_candidates_retrained": 26,
            "training_evidence_sha256": "0" * 64,
            "epoch_lineage_ledger_sha256": "0" * 64,
            "terminal_model_sha256": "0" * 64,
            "terminal_optimizer_sha256": "0" * 64,
            "terminal_rng_sha256": "0" * 64,
            "validation_selected_sha256": "0" * 64,
            "maximum_numeric_absolute_error": 0.0,
            "all_epoch_metrics_candidates_and_terminal_exact": True,
        }
        for strategy in ("mf", "nmf")
    ]
    sentinel_retrain_audit = {
        "schema_version": "t9.1.3-family-sentinel-retrain-v1",
        "claim_scope": "two_of_forty_production_agents",
        "sentinel_rule": "first_frozen_paired_root_per_family",
        "execution_mode": "sequential_same_runtime_agent_local_rng_isolation",
        "sentinel_agent_count": 2,
        "population_agent_count": 40,
        "families": sentinel_families,
        "all_sentinels_exactly_reproduced": True,
        "temporary_artifacts_deleted": True,
        "replay_ledger_sha256": subject._canonical_sha256(sentinel_families),
    }
    assert subject._sentinel_retrain_audit_valid(
        sentinel_retrain_audit,
        expected_epochs=1_000,
        expected_population_agents=40,
        expected_roots={"mf": int(roots[0]), "nmf": int(roots[0])},
    )
    paths = set(subject.MANDATORY_SHARED_ARTIFACT_PATHS)
    paths.update(
        f"docs/t9_1_3_puviani_agents/{strategy}/{strategy}_seed_{int(seed)}{suffix}"
        for strategy in ("mf", "nmf")
        for seed in roots
        for suffix in (".npz", ".terminal.pt", ".json")
    )
    entries = [
        {"path": path, "sha256": "0" * 64, "bytes": 0}
        for path in sorted(paths)
    ]
    selected = {
        strategy: {
            "selected_root_seed": int(roots[0]),
            "selected_checkpoint_epoch": 0,
            "selected_validation_score": 0.0,
            "checkpoint_sha256": "0" * 64,
            "agent_count": 20,
            "validation_candidates_per_agent": 26,
            "total_agent_checkpoint_candidates": 520,
            "all_agent_best_validation_score_distribution": subject._distribution(
                [0.0] * 20
            ),
            "all_agent_initial_validation_score_distribution": subject._distribution(
                [0.0] * 20
            ),
            "all_agent_initial_to_best_gain_distribution": subject._distribution(
                [0.0] * 20
            ),
            "all_agent_selected_checkpoint_epoch_distribution": subject._distribution(
                [0.0] * 20
            ),
            "selection_optimism_warning": (
                "best-of-20 agents and best-of-validation-checkpoints can be "
                "optimistic; these validation scores are selection diagnostics, "
                "not an unbiased performance estimate or a T9.1.3 performance gate; "
                "six-state qualification seeds are held out"
            ),
        }
        for strategy in ("mf", "nmf")
    }
    report: dict[str, object] = {
        "schema_version": subject.SCHEMA_VERSION,
        "task_id": subject.TASK_ID,
        "generated_at_utc": "2026-07-22T00:00:00+00:00",
        "status": subject.STATUS_PASS,
        "production_contract": True,
        "evidence_grade": subject.EVIDENCE_GRADE,
        "namespace": {
            "official_exact": {"state": "MISSING_EXTERNAL_ASSET", "value": None},
            "paper_constrained": "THIS_ARTIFACT",
            "project_native_directional": "T2.3.7_SEPARATE_NONTRANSFERABLE",
        },
        "t9_1_1_binding": parent,
        "implementation_sha256": subject.implementation_sha256(),
        "config": config,
        "config_sha256": subject._canonical_sha256(config),
        "source_registry": sources,
        "runtime_profile": runtime,
        "training_runtime_contract": training_runtime_contract,
        "protocol_transcription": deepcopy(config["protocol_transcription"]),
        "action_contract": {
            "project_order": list(subject.PROJECT_ACTION_ORDER),
            "gqf_order": list(subject.GQF_ACTION_ORDER),
            "permutation_roundtrip": True,
            "bounds": "14 residuals in [-2,2], virtual rotation in [-1,1] after tanh",
            "ranking_eligibility": config["ranking_eligibility"],
        },
        "selection_contract": {
            "within_agent": "max validation score; tie earlier epoch",
            "across_agents": (
                "max best-validation score; tie earlier epoch then lower root seed"
            ),
            "formal_used_for_selection": False,
            "test_used_for_selection": False,
            "paper_best_of_20_test_selection_replaced": True,
            "selection_lock_sha256": subject._canonical_sha256([]),
            "ledger_path": "docs/t9_1_3_puviani_selection_ledger.csv",
            "selected": selected,
        },
        "population": {
            "paired_root_count": 20,
            "total_agent_count": 40,
            "agents_per_family": {"mf": 20, "nmf": 20},
            "all_agents_published_not_only_winner": True,
            "agent_registry": "docs/t9_1_3_puviani_agent_registry.csv",
        },
        "training_ledger": {
            "path": "docs/t9_1_3_puviani_training_ledger.parquet",
            "rows": 40_000,
            "all_epochs_all_agents": True,
            "family_sentinel_full_retrain": sentinel_retrain_audit,
        },
        "six_state_evaluator": {
            "state_labels": list(STATE_LABELS),
            "methods": ["standard", "mf", "nmf"],
            "trajectory_rows": 1_008,
            "event_rows": 20_160,
            "trajectory_path": "docs/t9_1_3_puviani_six_state_trajectories.parquet",
            "event_path": "docs/t9_1_3_puviani_six_state_events.parquet",
            "postselection": False,
            "all_trajectories_retained": True,
            "development_only": True,
            "state_definition_id": "LOWDIN_ORTHONORMALIZED_FINITE_CUTOFF_GRID_CODE_V1",
            "logical_operator_definition_id": "PROJECTOR_DEFINED_CODE_PAULI_V1",
            "monte_carlo_design": (
                "paired_common_random_numbers_across_states_and_strategies_"
                "clustered_by_stratum_and_evaluation_seed"
            ),
            "reconstruction": reconstruction,
            "raw_replay_audit": {
                "schema_version": "t9.1.3-six-state-same-backend-replay-v1",
                "scope": (
                    "seeded same-runtime/same-backend replay of selected checkpoints, "
                    "RNG, physics, actions and raw rows; not cross-runtime bitwise "
                    "reproducibility and not an independent second physics backend"
                ),
                "group_count": 3
                * len(STATE_LABELS)
                * (
                    len(config["six_state_qualification"]["seeds"])
                    + len(config["six_state_qualification"]["confirmation_seeds"])
                ),
                "trajectory_count": 1_008,
                "event_count": 20_160,
                "maximum_action_absolute_error": 0.0,
                "maximum_branch_probability_absolute_error": 0.0,
                "maximum_projected_rho_absolute_error": 0.0,
                "all_rows_replayed": True,
            },
            "qualification_scope": (
                "finite-cutoff ten-cycle executable/raw qualification only; not a "
                "paper lifetime and not Phase9 >=1e4-cycle formal evidence"
            ),
        },
        "paper_scale_numerical_result": {
            "state": subject.PAPER_NUMERIC_STATE,
            "value": None,
            "required_cutoff": 100,
            "required_cycles": 1000,
            "current_qualification_cutoffs": [12, 16],
            "current_qualification_cycles": 10,
            "published_anchor_comparison_eligible": False,
            "reason": subject.PAPER_NUMERIC_REASON,
        },
        "external_claim_slots": {
            "official_exact": None,
            "puviani_surpass": None,
            "paper_T_X_T_Y_T_Z_T_ch": None,
        },
        "gates": {name: True for name in subject.REQUIRED_GATE_NAMES},
        "required_gates": list(subject.REQUIRED_GATE_NAMES),
        "artifact_manifest": {
            "files": entries,
            "sorted_manifest_sha256": subject._canonical_sha256(entries),
            "file_count": len(entries),
            "total_bytes": 0,
        },
        "failure_attempt_ledger": {
            "count": 0,
            "attempts": [],
            "distinct_runtime_signature_sha256": [],
            "all_failure_runtime_signatures_self_bound": True,
            "silent_drop": False,
            "note": (
                "A retained failed attempt does not replace the required completed agent; "
                "all 40 completed checkpoints remain mandatory."
            ),
        },
        "output_tree_audit": {
            "root": "docs/t9_1_3_puviani_agents",
            "output_file_count": 120,
            "unknown_or_orphan_files": [],
            "training_attempt_count": 0,
            "training_attempts": [],
            "all_training_attempts_terminal_and_valid": True,
            "every_agent_has_successful_attempt": True,
        },
        "deviation_ledger": {
            "path": str(config["deviation_ledger_path"]),
            "sha256": subject._file_sha256(deviation_path),
            "entry_count": len(deviations),
            "critical_or_high_count": sum(
                row["severity"] in {"High", "Critical"} for row in deviations
            ),
        },
        "claim_boundary": dict(subject.REPORT_CLAIM_BOUNDARY),
        "wall_time_seconds_finalize": 0.0,
    }
    report["mutation_audit"] = subject._mutation_audit(report)
    return _reseal(report)


def test_action_permutation_roundtrip_and_shape_rejection() -> None:
    project = np.arange(30, dtype=np.float64).reshape(2, 15)
    public_gqf = subject.project_to_gqf_action(project)
    assert public_gqf.shape == project.shape
    assert np.array_equal(public_gqf[:, 0], project[:, 14])
    assert np.array_equal(subject.gqf_to_project_action(public_gqf), project)

    for bad in (np.zeros(14), np.zeros((2, 14)), np.zeros((2, 16))):
        with pytest.raises(ValueError, match="15 parameters"):
            subject.project_to_gqf_action(bad)
        with pytest.raises(ValueError, match="15 parameters"):
            subject.gqf_to_project_action(bad)


def test_production_config_passes_live_fail_closed_validation() -> None:
    config = _load_production_config()
    subject._validate_config(config, production=True)
    parent = subject._verify_parent_protocol(config)

    training = config["training"]
    qualification = config["six_state_qualification"]
    artifact = config["artifact_contract"]
    assert isinstance(training, dict)
    assert isinstance(qualification, dict)
    assert isinstance(artifact, dict)
    assert parent["analysis_sha256"] == subject.T9_1_1_ANALYSIS_SHA256
    assert len(training["paired_root_seeds"]) == 20
    assert artifact["agent_manifest_rows_expected"] == 40
    assert training["epochs"] == 1000
    assert tuple(qualification["states"]) == STATE_LABELS
    assert qualification["full_cycles"] == 10


def test_production_runtime_contract_rejects_missing_packages_cuda_and_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_production_config()
    signature, _ = subject._current_training_runtime()
    signature = _production_deterministic_signature(signature)

    def install_runtime(value: dict[str, object]) -> None:
        normalized, digest = subject._validated_training_runtime_signature(
            value, label="synthetic runtime"
        )
        monkeypatch.setattr(
            subject,
            "_current_training_runtime",
            lambda: (deepcopy(normalized), digest),
        )

    missing_package = deepcopy(signature)
    missing_package["pyarrow_version"] = None
    install_runtime(missing_package)
    with pytest.raises(RuntimeError, match="required packages: pyarrow_version"):
        subject._validate_runtime_for_config(config, production=True)

    cpu_only = deepcopy(signature)
    cpu_only.update(
        {
            "torch_cuda_version": None,
            "cudnn_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_current_device": None,
            "cuda_device_names": [],
            "cuda_device_capabilities": [],
            "cuda_total_memory_bytes": [],
            "nvidia_smi_devices": [],
        }
    )
    install_runtime(cpu_only)
    with pytest.raises(RuntimeError, match="live CUDA torch runtime"):
        subject._validate_runtime_for_config(config, production=True)
    # The same complete CPU signature is legal for an explicitly nonproduction
    # tiny run; this keeps the negative test from encoding a CUDA-everywhere rule.
    assert subject._validate_runtime_for_config(config, production=False)[0] == cpu_only

    no_driver = deepcopy(signature)
    no_driver["nvidia_smi_devices"] = []
    install_runtime(no_driver)
    with pytest.raises(RuntimeError, match="NVIDIA UUID/driver ledger"):
        subject._validate_runtime_for_config(config, production=True)


def test_production_determinism_policy_is_exact_and_cpu_testable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature, _ = subject._current_training_runtime()
    good = _production_deterministic_signature(signature)
    subject._validate_production_determinism_signature(good)
    for field, bad in (
        ("deterministic_algorithms_enabled", False),
        ("deterministic_debug_mode", 1),
        ("cudnn_deterministic", False),
        ("cudnn_benchmark", True),
        ("cudnn_allow_tf32", True),
        ("cuda_matmul_allow_tf32", True),
        ("float32_matmul_precision", "high"),
    ):
        changed = deepcopy(good)
        changed[field] = bad
        with pytest.raises(RuntimeError, match="deterministic/TF32"):
            subject._validate_production_determinism_signature(changed)
    changed = deepcopy(good)
    changed["environment_controls"]["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    with pytest.raises(RuntimeError, match="environment controls drifted"):
        subject._validate_production_determinism_signature(changed)

    calls: dict[str, object] = {}

    class FakeTorch:
        backends = SimpleNamespace(
            cudnn=SimpleNamespace(
                deterministic=False, benchmark=True, allow_tf32=True
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )

        @staticmethod
        def use_deterministic_algorithms(enabled: bool, *, warn_only: bool) -> None:
            calls["algorithms"] = (enabled, warn_only)

        @staticmethod
        def set_float32_matmul_precision(value: str) -> None:
            calls["precision"] = value

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(subject, "_require_torch", lambda: FakeTorch)
    subject._configure_production_determinism()
    assert calls == {"algorithms": (True, False), "precision": "highest"}
    assert FakeTorch.backends.cudnn.deterministic is True
    assert FakeTorch.backends.cudnn.benchmark is False
    assert FakeTorch.backends.cudnn.allow_tf32 is False
    assert FakeTorch.backends.cuda.matmul.allow_tf32 is False


def test_frozen_config_nonproduction_api_downgrade_rejected_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load_production_config()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("side effect occurred before downgrade rejection")

    monkeypatch.setattr(subject, "_require_torch", forbidden)
    monkeypatch.setattr(subject, "_namespace_finalize_lock", forbidden)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="cannot run with production=False"):
        subject.train_population(
            config, output_dir=output, family="mf", production=False
        )
    common = {
        "output_dir": output,
        "report_path": tmp_path / "report.json",
        "agent_registry_path": tmp_path / "agents.csv",
        "selection_ledger_path": tmp_path / "selection.csv",
        "training_ledger_path": tmp_path / "training.parquet",
        "trajectory_path": tmp_path / "trajectories.parquet",
        "event_path": tmp_path / "events.parquet",
        "production": False,
    }
    with pytest.raises(ValueError, match="cannot run with production=False"):
        subject.finalize_artifacts(config, **common)
    with pytest.raises(ValueError, match="cannot run with production=False"):
        subject._finalize_artifacts_locked(config, **common)
    assert not output.exists()


def test_runtime_signature_is_part_of_agent_resume_identity() -> None:
    config = _pilot_config()
    signature, _ = subject._current_training_runtime()
    changed = deepcopy(signature)
    changed["python_version"] += "-different-runtime"
    first = subject._agent_contract_hash(
        config,
        "mf",
        991_001,
        subject.implementation_sha256(),
        signature,
    )
    second = subject._agent_contract_hash(
        config,
        "mf",
        991_001,
        subject.implementation_sha256(),
        changed,
    )
    assert first != second


def test_runtime_signature_is_uncached_across_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = subject._training_runtime_signature()
    monkeypatch.setenv("PYTHONHASHSEED", "t913-runtime-mutation-test")
    after = subject._training_runtime_signature()
    assert before["environment_controls"]["PYTHONHASHSEED"] != (
        after["environment_controls"]["PYTHONHASHSEED"]
    )
    assert after["environment_controls"]["PYTHONHASHSEED"] == (
        "t913-runtime-mutation-test"
    )


def test_training_runtime_contract_detects_mixed_agent_environments() -> None:
    signature, digest = subject._current_training_runtime()
    runtime_profile = {
        "signature_sha256": digest,
    }
    identical_agents = {
        strategy: [
            (
                object(),
                {
                    "training_runtime_signature": deepcopy(signature),
                    "training_runtime_signature_sha256": digest,
                },
            )
        ]
        for strategy in ("mf", "nmf")
    }
    identical = subject._training_runtime_contract(
        identical_agents, runtime_profile, production=False
    )
    assert identical["agent_signature_count"] == 2
    assert identical["all_agents_identical"] is True
    assert identical["equals_finalizer_runtime_signature"] is True

    changed = deepcopy(signature)
    changed["python_version"] += "-other-agent"
    changed, changed_digest = subject._validated_training_runtime_signature(
        changed, label="different agent runtime"
    )
    mixed_agents = deepcopy(identical_agents)
    mixed_agents["nmf"][0][1]["training_runtime_signature"] = changed
    mixed_agents["nmf"][0][1][
        "training_runtime_signature_sha256"
    ] = changed_digest
    mixed = subject._training_runtime_contract(
        mixed_agents, runtime_profile, production=False
    )
    assert mixed["agent_signature_count"] == 2
    assert mixed["signature"] is None
    assert mixed["signature_sha256"] is None
    assert mixed["all_agents_identical"] is False
    assert mixed["equals_finalizer_runtime_signature"] is False


def test_training_runtime_contract_enforces_determinism_only_in_production() -> None:
    signature, _ = subject._current_training_runtime()
    nondeterministic = deepcopy(signature)
    nondeterministic["deterministic_algorithms_enabled"] = False
    nondeterministic["deterministic_debug_mode"] = 0
    nondeterministic["environment_controls"]["CUBLAS_WORKSPACE_CONFIG"] = None
    nondeterministic, digest = subject._validated_training_runtime_signature(
        nondeterministic, label="nonproduction runtime"
    )
    agents = {
        strategy: [
            (
                object(),
                {
                    "training_runtime_signature": deepcopy(nondeterministic),
                    "training_runtime_signature_sha256": digest,
                },
            )
        ]
        for strategy in ("mf", "nmf")
    }
    profile = {"signature_sha256": digest}
    pilot = subject._training_runtime_contract(
        agents, profile, production=False
    )
    assert pilot["all_agents_identical"] is True
    with pytest.raises(RuntimeError, match="deterministic environment controls"):
        subject._training_runtime_contract(agents, profile, production=True)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda cfg: cfg["training"].update(
            {"epochs_by_strategy": {"mf": 1, "nmf": 2}}
        ),
        lambda cfg: cfg["training"]["selection"].update(
            {"selection_data": "test"}
        ),
        lambda cfg: cfg["training"]["checkpoint_contract"].update(
            {"required_per_agent": []}
        ),
        lambda cfg: cfg["six_state_qualification"].update(
            {"postselection": "ALLOWED"}
        ),
        lambda cfg: cfg["six_state_qualification"].update(
            {"primary_cutoff": 11}
        ),
        lambda cfg: cfg["paper_scale_target"]["numeric_values"].update(
            {"T_X": 999.0}
        ),
        lambda cfg: cfg["external_claim_slots"]["surpass"].update(
            {"value": {"gain": 1.1}}
        ),
    ),
)
def test_production_config_rejects_claim_and_selection_contract_mutations(
    mutate: object,
) -> None:
    config = _load_production_config()
    mutate(config)
    with pytest.raises(ValueError):
        subject._validate_config(config, production=True)


@pytest.mark.parametrize("strategy", ("mf", "nmf"))
def test_tiny_cpu_training_persists_restart_state_reloads_npz_and_rejects_drift(
    tmp_path: Path,
    strategy: str,
) -> None:
    torch = pytest.importorskip("torch")
    config = _pilot_config()
    output = tmp_path / "agents"
    first = subject.train_population(
        config,
        output_dir=output,
        family=strategy,
        production=False,
    )
    assert first["requested_agents"] == 1
    assert first["newly_trained_agents"] == 1
    assert first["resumed_agents"] == 0

    root_seed = int(config["training"]["paired_root_seeds"][0])
    weight_path, terminal_path, record_path = subject._agent_paths(
        output, strategy, root_seed
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert weight_path.is_file() and terminal_path.is_file()
    assert record["training_record"]["epochs_executed"] == 1
    assert len(record["training_record"]["training_curve"]) == 1
    assert record["reload_witness"] is True
    assert record["terminal_reload_witness"] is True
    assert record["validation_candidate_reload_witness"] is True
    assert record["attempt_id"]
    runtime_signature, runtime_hash = subject._current_training_runtime()
    assert record["training_runtime_signature"] == runtime_signature
    assert record["training_runtime_signature_sha256"] == runtime_hash
    assert subject._canonical_sha256(
        record["strategy_config"]
    ) == subject._canonical_sha256(
        subject.asdict(subject._strategy_config(config, strategy))
    )

    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    assert terminal["terminal_epoch"] == 1
    assert terminal["optimizer_state"]["state"]
    assert terminal["optimizer_state"]["param_groups"]
    assert terminal["torch_cpu_rng_state"].numel() > 0
    assert "numpy_random_state" in terminal
    assert terminal["config_sha256"] == subject._canonical_sha256(config)
    assert terminal["implementation_sha256"] == subject.implementation_sha256()
    assert terminal["attempt_id"] == record["attempt_id"]
    assert terminal["training_runtime_signature"] == runtime_signature
    assert terminal["training_runtime_signature_sha256"] == runtime_hash

    attempts = list(
        (output / "_training_attempts").glob(
            f"{strategy}_seed_{root_seed}_*"
        )
    )
    assert len(attempts) == 1
    start = json.loads((attempts[0] / "start.json").read_text(encoding="utf-8"))
    outcome = json.loads(
        (attempts[0] / "outcome.json").read_text(encoding="utf-8")
    )
    assert start["runtime_signature_state"] == "CAPTURED_BEFORE_TRAINING"
    assert start["training_runtime_signature"] == runtime_signature
    assert start["training_runtime_signature_sha256"] == runtime_hash
    assert outcome["training_runtime_signature_sha256"] == runtime_hash

    validation_history = record["training_record"]["validation_history"]
    candidates = terminal["validation_candidate_states"]
    assert [row["epoch"] for row in validation_history] == [0, 1]
    assert [row["epoch"] for row in candidates] == [0, 1]
    assert len(candidates) == len(validation_history)
    assert record["training_record"][
        "all_validation_candidate_states_retained"
    ] is True
    assert record["training_record"][
        "validation_raw_per_seed_and_trajectory_retained"
    ] is True
    for history_row, candidate in zip(validation_history, candidates, strict=True):
        assert candidate["state_sha256"] == history_row["candidate_state_sha256"]
        assert subject.state_dict_sha256(candidate["model_state"]) == candidate[
            "state_sha256"
        ]
        raw = history_row["validation_raw_per_seed"]
        assert [row["seed"] for row in raw] == list(
            config["training"]["validation_seeds"]
        )
        assert all(
            len(row["trajectory_final_fidelity"])
            == config["training"]["validation_batch_size"]
            for row in raw
        )

    directional = subject._strategy_config(config, strategy)
    reloaded_model, tensor_hash = subject._load_npz_model(
        weight_path, strategy, directional, root_seed
    )
    assert reloaded_model.parameter_count == (
        70_159 if strategy == "mf" else 72_853
    )
    assert tensor_hash == record["canonical_tensor_sha256"]

    # A coordinated raw/score/winner re-seal is internally self-consistent,
    # so reload/hash checks alone cannot detect it.  The full candidate physics
    # replay must be the control that closes this forgery path.
    forged_training = deepcopy(record["training_record"])
    forged_terminal = deepcopy(terminal)
    for history_row in forged_training["validation_history"]:
        for raw in history_row["validation_raw_per_seed"]:
            raw["trajectory_final_fidelity"] = [
                0.5 * float(value)
                for value in raw["trajectory_final_fidelity"]
            ]
            raw["fidelity_curve"][-1] = float(
                np.mean(raw["trajectory_final_fidelity"])
            )
        history_row["selection_score"] = subject._validation_final_fidelity(
            {"per_seed": history_row["validation_raw_per_seed"]}
        )
    forged_best = sorted(
        forged_training["validation_history"],
        key=lambda row: (-float(row["selection_score"]), int(row["epoch"])),
    )[0]
    forged_training["best_validation_epoch"] = int(forged_best["epoch"])
    forged_training["best_validation_score"] = float(
        forged_best["selection_score"]
    )
    subject._validate_validation_evidence(
        training_record=forged_training,
        terminal=forged_terminal,
        model=reloaded_model,
        directional=directional,
        strategy=strategy,
        seed=root_seed,
        replay_physics=False,
    )
    with pytest.raises(
        subject.CorruptAgentArtifactError,
        match="not reproducible|physics-reproducible",
    ):
        subject._validate_validation_evidence(
            training_record=forged_training,
            terminal=forged_terminal,
            model=reloaded_model,
            directional=directional,
            strategy=strategy,
            seed=root_seed,
            replay_physics=True,
        )

    loaded = subject._load_agent_record(
        output_dir=output,
        strategy=strategy,
        seed=root_seed,
        directional=directional,
        config=config,
        implementation_hash=subject.implementation_sha256(),
        replay_validation_candidates=True,
    )
    assert loaded is not None
    replay = loaded[1]["live_validation_candidate_physics_replay"]
    assert replay["all_candidates_replayed"] is True
    assert replay["candidate_count"] == 2
    assert replay["validation_seed_count"] == 1
    assert replay["trajectories_per_candidate"] == 1
    assert replay["maximum_numeric_absolute_error"] <= 2.0e-10
    _, ledger = subject._select_agents({"mf": [loaded], "nmf": [loaded]})
    strategy_rows = [row for row in ledger if row["strategy"] == strategy]
    assert sum(row["selected_within_agent"] for row in strategy_rows) == 1
    assert sum(row["selected_across_agents"] for row in strategy_rows) == 1
    assert [row["checkpoint_tensor_sha256"] for row in strategy_rows] == [
        point["candidate_state_sha256"]
        for point in record["training_record"]["validation_history"]
    ]

    resumed = subject.train_population(
        config,
        output_dir=output,
        family=strategy,
        production=False,
    )
    assert resumed["newly_trained_agents"] == 0
    assert resumed["resumed_agents"] == 1

    changed = deepcopy(config)
    changed["training"]["learning_rate"] = 2.0e-4
    with pytest.raises(ValueError, match="agent contract drift"):
        subject.train_population(
            changed,
            output_dir=output,
            family=strategy,
            production=False,
        )

    # Every persisted runtime copy is equality-bound, not merely present.
    pristine_record = json.loads(record_path.read_text(encoding="utf-8"))
    pristine_terminal = torch.load(
        terminal_path, map_location="cpu", weights_only=False
    )

    bad_record = deepcopy(pristine_record)
    bad_record["training_runtime_signature"]["python_version"] += "-tampered"
    subject._atomic_json(bad_record, record_path)
    with pytest.raises(
        subject.CorruptAgentArtifactError, match="runtime signature hash"
    ):
        subject._load_agent_record(
            output_dir=output,
            strategy=strategy,
            seed=root_seed,
            directional=directional,
            config=config,
            implementation_hash=subject.implementation_sha256(),
        )
    subject._atomic_json(pristine_record, record_path)

    bad_terminal = deepcopy(pristine_terminal)
    bad_terminal["training_runtime_signature"]["python_version"] += "-mixed"
    bad_terminal["training_runtime_signature_sha256"] = subject._canonical_sha256(
        bad_terminal["training_runtime_signature"]
    )
    subject._atomic_torch(bad_terminal, terminal_path)
    bad_record = deepcopy(pristine_record)
    bad_record["terminal_file_sha256"] = subject._file_sha256(terminal_path)
    bad_record["terminal_bytes"] = terminal_path.stat().st_size
    subject._atomic_json(bad_record, record_path)
    with pytest.raises(
        subject.CorruptAgentArtifactError, match="terminal/agent runtime signature"
    ):
        subject._load_agent_record(
            output_dir=output,
            strategy=strategy,
            seed=root_seed,
            directional=directional,
            config=config,
            implementation_hash=subject.implementation_sha256(),
        )
    subject._atomic_torch(pristine_terminal, terminal_path)
    pristine_record["terminal_file_sha256"] = subject._file_sha256(terminal_path)
    pristine_record["terminal_bytes"] = terminal_path.stat().st_size
    outcome["terminal_file_sha256"] = pristine_record["terminal_file_sha256"]
    subject._atomic_json(pristine_record, record_path)
    subject._atomic_json(outcome, attempts[0] / "outcome.json")

    bad_outcome = deepcopy(outcome)
    bad_outcome["training_runtime_signature_sha256"] = "f" * 64
    subject._atomic_json(bad_outcome, attempts[0] / "outcome.json")
    audit = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert audit["all_training_attempts_terminal_and_valid"] is False
    assert audit["training_attempts"][0]["valid_outcome"] is False
    subject._atomic_json(outcome, attempts[0] / "outcome.json")

    # JSON ledgers reject both NaN and infinity even when their container and
    # file hashes remain otherwise well formed.
    for nonfinite in (float("nan"), float("inf")):
        bad_record = deepcopy(pristine_record)
        bad_record["training_record"]["training_curve"][0]["loss"] = nonfinite
        subject._atomic_json(bad_record, record_path)
        with pytest.raises(
            subject.CorruptAgentArtifactError,
            match="training/validation ledger mismatch",
        ):
            subject._load_agent_record(
                output_dir=output,
                strategy=strategy,
                seed=root_seed,
                directional=directional,
                config=config,
                implementation_hash=subject.implementation_sha256(),
            )
    subject._atomic_json(pristine_record, record_path)

    bad_record = deepcopy(pristine_record)
    bad_record["strategy_config"]["learning_rate"] *= 2.0
    subject._atomic_json(bad_record, record_path)
    with pytest.raises(
        subject.CorruptAgentArtifactError, match="provenance mismatch"
    ):
        subject._load_agent_record(
            output_dir=output,
            strategy=strategy,
            seed=root_seed,
            directional=directional,
            config=config,
            implementation_hash=subject.implementation_sha256(),
        )
    subject._atomic_json(pristine_record, record_path)

    bad_terminal = deepcopy(pristine_terminal)
    bad_terminal["validation_candidate_states"][-1]["state_sha256"] = "f" * 64
    subject._atomic_torch(bad_terminal, terminal_path)
    bad_record = deepcopy(pristine_record)
    bad_record["terminal_file_sha256"] = subject._file_sha256(terminal_path)
    bad_record["terminal_bytes"] = terminal_path.stat().st_size
    subject._atomic_json(bad_record, record_path)
    with pytest.raises(
        subject.CorruptAgentArtifactError, match="validation candidate hash drift"
    ):
        subject._load_agent_record(
            output_dir=output,
            strategy=strategy,
            seed=root_seed,
            directional=directional,
            config=config,
            implementation_hash=subject.implementation_sha256(),
        )
    subject._atomic_torch(pristine_terminal, terminal_path)
    pristine_record["terminal_file_sha256"] = subject._file_sha256(terminal_path)
    pristine_record["terminal_bytes"] = terminal_path.stat().st_size
    outcome["terminal_file_sha256"] = pristine_record["terminal_file_sha256"]
    subject._atomic_json(pristine_record, record_path)
    subject._atomic_json(outcome, attempts[0] / "outcome.json")

    # Rebuild an exact validation-score tie and prove that the earlier epoch is
    # the sole legal winner.  This exercises the tie rule independently of the
    # random tiny run's observed ordering.
    tied_record = deepcopy(pristine_record["training_record"])
    tied_terminal = deepcopy(pristine_terminal)
    first_validation = tied_record["validation_history"][0]
    last_validation = tied_record["validation_history"][-1]
    last_validation["selection_score"] = first_validation["selection_score"]
    last_validation["metric_means"] = deepcopy(first_validation["metric_means"])
    last_validation["validation_raw_per_seed"] = deepcopy(
        first_validation["validation_raw_per_seed"]
    )
    first_validation["selected_when_observed"] = True
    last_validation["selected_when_observed"] = False
    first_validation["selected_final"] = True
    last_validation["selected_final"] = False
    tied_record["best_validation_epoch"] = int(first_validation["epoch"])
    tied_record["best_validation_score"] = float(
        first_validation["selection_score"]
    )
    subject._validate_validation_evidence(
        training_record=tied_record,
        terminal=tied_terminal,
        model=reloaded_model,
        directional=directional,
        strategy=strategy,
        seed=root_seed,
    )
    tied_record["best_validation_epoch"] = int(last_validation["epoch"])
    first_validation["selected_final"] = False
    last_validation["selected_final"] = True
    with pytest.raises(
        subject.CorruptAgentArtifactError, match="validation best/tie contract"
    ):
        subject._validate_validation_evidence(
            training_record=tied_record,
            terminal=tied_terminal,
            model=reloaded_model,
            directional=directional,
            strategy=strategy,
            seed=root_seed,
        )

    # Re-hashing a structurally complete but semantically invalid restart file
    # must not bypass the RNG loader checks.
    terminal["numpy_random_state"] = ("BROKEN_RNG_STATE",)
    subject._atomic_torch(terminal, terminal_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["terminal_file_sha256"] = subject._file_sha256(terminal_path)
    record["terminal_bytes"] = terminal_path.stat().st_size
    subject._atomic_json(record, record_path)
    with pytest.raises(subject.CorruptAgentArtifactError, match="optimizer/RNG"):
        subject._load_agent_record(
            output_dir=output,
            strategy=strategy,
            seed=root_seed,
            directional=directional,
            config=config,
            implementation_hash=subject.implementation_sha256(),
        )


def test_agent_failure_is_atomically_retained_before_fail_closed_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pilot_config()

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("non-finite synthetic training loss")

    monkeypatch.setattr(subject, "_train_paper_agent", fail)
    with pytest.raises(RuntimeError, match="non-finite synthetic"):
        subject.train_population(
            config,
            output_dir=tmp_path / "failed_agents",
            family="mf",
            production=False,
        )
    root_seed = int(config["training"]["paired_root_seeds"][0])
    failure_paths = list((tmp_path / "failed_agents").rglob("failure.json"))
    assert len(failure_paths) == 1
    failure_path = failure_paths[0]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["failure_state"] == "NUMERIC_FAILURE"
    assert failure["retained_instead_of_silent_drop"] is True
    assert failure["root_seed"] == root_seed
    runtime_signature, runtime_hash = subject._current_training_runtime()
    assert failure["training_runtime_signature"] == runtime_signature
    assert failure["training_runtime_signature_sha256"] == runtime_hash
    assert failure["agent_contract_hash"] == subject._agent_contract_hash(
        config,
        "mf",
        root_seed,
        subject.implementation_sha256(),
        runtime_signature,
    )
    output = tmp_path / "failed_agents"
    audit = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert audit["training_attempts"][0]["valid_start"] is True
    assert audit["training_attempts"][0]["valid_outcome"] is True
    assert audit["all_training_attempts_terminal_and_valid"] is True

    # Both start and failure can remain individually well-formed while being
    # drawn from different runtime lineages.  Their attempt binding must fail.
    attempt = next((output / "_training_attempts").iterdir())
    start_path = attempt / "start.json"
    start = json.loads(start_path.read_text(encoding="utf-8"))
    other_runtime = deepcopy(start["training_runtime_signature"])
    other_runtime["python_version"] += "-other-start-runtime"
    other_hash = subject._canonical_sha256(other_runtime)
    start["training_runtime_signature"] = other_runtime
    start["training_runtime_signature_sha256"] = other_hash
    start["agent_contract_hash"] = subject._agent_contract_hash(
        config,
        "mf",
        root_seed,
        subject.implementation_sha256(),
        other_runtime,
    )
    subject._atomic_json(start, start_path)
    mixed = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert mixed["training_attempts"][0]["valid_start"] is True
    assert mixed["training_attempts"][0]["valid_outcome"] is False
    assert mixed["all_training_attempts_terminal_and_valid"] is False


def test_partial_bundle_is_archived_before_retraining_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pilot_config()
    output = tmp_path / "partial_agents"
    root_seed = int(config["training"]["paired_root_seeds"][0])
    weight_path, _, _ = subject._agent_paths(output, "mf", root_seed)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"interrupted-npz")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic retry stop")

    monkeypatch.setattr(subject, "_train_paper_agent", fail)
    with pytest.raises(RuntimeError, match="synthetic retry stop"):
        subject.train_population(
            config, output_dir=output, family="mf", production=False
        )
    failures = [json.loads(path.read_text(encoding="utf-8")) for path in output.rglob("failure.json")]
    assert {row["failure_state"] for row in failures} == {"CORRUPT_SHARD", "NULL"}
    corrupt = next(row for row in failures if row["failure_state"] == "CORRUPT_SHARD")
    assert len(corrupt["archived_partial_files"]) == 1
    archived = Path(corrupt["archived_partial_files"][0]["path"])
    assert archived.read_bytes() == b"interrupted-npz"
    assert not weight_path.exists()


def test_same_agent_writer_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    config = _pilot_config()
    seed = int(config["training"]["paired_root_seeds"][0])
    implementation_hash = subject.implementation_sha256()
    arguments = {
        "output_dir": tmp_path / "locked",
        "strategy": "mf",
        "seed": seed,
        "config": config,
        "implementation_hash": implementation_hash,
    }
    with subject._agent_writer_lock(**arguments):
        lock_dir = tmp_path / "locked" / "_locks" / f"mf_seed_{seed}.lock"
        assert lock_dir.is_dir()
        assert (lock_dir / "owner.json").is_file()
        with pytest.raises(subject.ConcurrentAgentWriterError, match="live writer"):
            with subject._agent_writer_lock(**arguments):
                raise AssertionError("second writer should never enter")
    assert not lock_dir.exists()


def test_dead_local_writer_lock_is_retained_and_immediately_recovered(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    seed = int(config["training"]["paired_root_seeds"][0])
    output = tmp_path / "dead_lock"
    lock_dir = output / "_locks" / f"mf_seed_{seed}.lock"
    lock_dir.mkdir(parents=True)
    subject._atomic_json(
        {
            "schema_version": "t9.1.3-writer-lock-v1",
            "task_id": subject.TASK_ID,
            "strategy": "mf",
            "root_seed": seed,
            "pid": 2_147_483_647,
            "hostname": socket.gethostname(),
            "token": "dead-owner",
            "agent_contract_hash": subject._agent_contract_hash(
                config, "mf", seed, subject.implementation_sha256()
            ),
            "started_at_utc": "2000-01-01T00:00:00+00:00",
        },
        lock_dir / "owner.json",
    )
    with subject._agent_writer_lock(
        output_dir=output,
        strategy="mf",
        seed=seed,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    ):
        assert lock_dir.is_dir()
    failures = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.rglob("failure.json")
    ]
    assert len(failures) == 1
    assert failures[0]["failure_subtype"] == "STALE_WRITER_LOCK"
    assert failures[0]["stale_lock_metadata"]["token"] == "dead-owner"


def test_pid_reuse_identity_does_not_keep_a_dead_writer_lock_alive(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    seed = int(config["training"]["paired_root_seeds"][0])
    output = tmp_path / "pid_reuse"
    lock_dir = output / "_locks" / f"mf_seed_{seed}.lock"
    lock_dir.mkdir(parents=True)
    subject._atomic_json(
        {
            "schema_version": "t9.1.3-writer-lock-v1",
            "task_id": subject.TASK_ID,
            "strategy": "mf",
            "root_seed": seed,
            "pid": subject.os.getpid(),
            "process_created_unix_ns": 1,
            "hostname": socket.gethostname(),
            "token": "reused-pid-owner",
            "agent_contract_hash": subject._agent_contract_hash(
                config, "mf", seed, subject.implementation_sha256()
            ),
            "started_at_utc": "2000-01-01T00:00:00+00:00",
            "last_heartbeat_unix_ns": time.time_ns(),
            "last_heartbeat_utc": "2000-01-01T00:00:00+00:00",
            "heartbeat_sequence": 10,
            "lease_seconds": subject.LOCK_LEASE_SECONDS,
        },
        lock_dir / "owner.json",
    )
    with subject._agent_writer_lock(
        output_dir=output,
        strategy="mf",
        seed=seed,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    ):
        current = json.loads((lock_dir / "owner.json").read_text(encoding="utf-8"))
        assert current["token"] != "reused-pid-owner"


def test_pid_reuse_identity_does_not_keep_an_interrupted_start_alive(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "start_pid_reuse"
    seed = int(config["training"]["paired_root_seeds"][0])
    implementation_hash = subject.implementation_sha256()
    runtime_signature, runtime_hash = subject._current_training_runtime()
    attempt = subject._new_attempt_directory(
        output, "_training_attempts", "mf", seed
    )
    subject._atomic_json(
        {
            "schema_version": "t9.1.3-training-attempt-start-v1",
            "task_id": subject.TASK_ID,
            "strategy": "mf",
            "root_seed": seed,
            "pid": subject.os.getpid(),
            "process_created_unix_ns": 1,
            "hostname": socket.gethostname(),
            "config_sha256": subject._canonical_sha256(config),
            "agent_contract_hash": subject._agent_contract_hash(
                config, "mf", seed, implementation_hash, runtime_signature
            ),
            "training_runtime_signature": runtime_signature,
            "training_runtime_signature_sha256": runtime_hash,
            "runtime_signature_state": "CAPTURED_BEFORE_TRAINING",
            "implementation_sha256": implementation_hash,
            "parent_analysis_sha256": subject.T9_1_1_ANALYSIS_SHA256,
            "attempt_id": attempt.name,
            "state": "STARTED",
            "started_at_utc": "2000-01-01T00:00:00+00:00",
        },
        attempt / "start.json",
    )
    subject._recover_interrupted_training_attempts(
        output_dir=output,
        strategy="mf",
        seed=seed,
        config=config,
        directional=subject._strategy_config(config, "mf"),
        implementation_hash=implementation_hash,
    )
    failure = json.loads((attempt / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_subtype"] == "INTERRUPTED_BEFORE_OUTCOME"
    assert failure["attempt_id"] == attempt.name


def test_cross_host_writer_is_fail_closed_even_after_heartbeat_lease_age(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    seed = int(config["training"]["paired_root_seeds"][0])
    output = tmp_path / "remote_lease"
    lock_dir = output / "_locks" / f"mf_seed_{seed}.lock"
    lock_dir.mkdir(parents=True)
    owner = {
        "schema_version": "t9.1.3-writer-lock-v1",
        "task_id": subject.TASK_ID,
        "strategy": "mf",
        "root_seed": seed,
        "pid": 123,
        "process_created_unix_ns": 123,
        "hostname": "different-host",
        "token": "remote-owner",
        "agent_contract_hash": subject._agent_contract_hash(
            config, "mf", seed, subject.implementation_sha256()
        ),
        "started_at_utc": "2000-01-01T00:00:00+00:00",
        "last_heartbeat_unix_ns": time.time_ns(),
        "last_heartbeat_utc": "now",
        "heartbeat_sequence": 99,
        "lease_seconds": subject.LOCK_LEASE_SECONDS,
    }
    subject._atomic_json(owner, lock_dir / "owner.json")
    old = time.time() - 10.0 * subject.LOCK_LEASE_SECONDS
    subject.os.utime(lock_dir, (old, old))
    with pytest.raises(subject.ConcurrentAgentWriterError, match="fail-closed"):
        with subject._agent_writer_lock(
            output_dir=output,
            strategy="mf",
            seed=seed,
            config=config,
            implementation_hash=subject.implementation_sha256(),
        ):
            raise AssertionError("fresh heartbeat must retain remote ownership")

    owner["last_heartbeat_unix_ns"] = time.time_ns() - int(
        2.0 * subject.LOCK_LEASE_SECONDS * 1_000_000_000
    )
    subject._atomic_json(owner, lock_dir / "owner.json")
    with pytest.raises(subject.ConcurrentAgentWriterError, match="fail-closed"):
        with subject._agent_writer_lock(
            output_dir=output,
            strategy="mf",
            seed=seed,
            config=config,
            implementation_hash=subject.implementation_sha256(),
        ):
            raise AssertionError("remote ownership requires explicit operator recovery")


def test_first_partial_archive_recovers_without_an_active_success(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "first_partial_archive"
    seed = int(config["training"]["paired_root_seeds"][0])
    archive = subject._new_attempt_directory(
        output, "_corrupt_attempts", "mf", seed
    )
    (archive / f"mf_seed_{seed}.npz").write_bytes(b"partial-first-publication")
    subject._recover_incomplete_archives(
        output, config, subject.implementation_sha256()
    )
    failure = json.loads((archive / "failure.json").read_text(encoding="utf-8"))
    assert failure["safe_retry_authorized"] is True
    assert failure["failure_subtype"] == "ARCHIVE_INTERRUPTED_AND_RECOVERED"
    assert subject._active_success_attempt_directories(output, "mf", seed) == []


def test_hard_kill_atomic_temp_is_quarantined_and_classified(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "atomic_kill"
    seed = int(config["training"]["paired_root_seeds"][0])
    weight_path = subject._agent_paths(output, "mf", seed)[0]
    weight_path.parent.mkdir(parents=True)
    temporary = weight_path.parent / f".{weight_path.name}.host.999.1.tmp"
    temporary.write_bytes(b"npz-write-killed-before-replace")
    failure_path = subject._quarantine_agent_atomic_temporaries(
        output_dir=output,
        strategy="mf",
        seed=seed,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert failure_path is not None and failure_path.is_file()
    assert not temporary.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    archived = Path(failure["archived_partial_files"][0]["path"])
    assert archived.read_bytes() == b"npz-write-killed-before-replace"
    audit = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert audit["unknown_or_orphan_files"] == []


def test_interrupted_stale_lock_archive_gets_a_recovery_marker(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "interrupted_stale_archive"
    seed = int(config["training"]["paired_root_seeds"][0])
    directory = subject._new_attempt_directory(
        output, "_failed_attempts", "mf", seed
    )
    stale = directory / "stale_writer_lock"
    stale.mkdir()
    (stale / "owner.json").write_text('{"interrupted":true}', encoding="utf-8")
    subject._recover_incomplete_failed_attempts(
        output, config, subject.implementation_sha256()
    )
    failure = json.loads((directory / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_subtype"] == "FAILED_ATTEMPT_ARCHIVE_INTERRUPTED_AND_RECOVERED"
    assert failure["safe_retry_authorized"] is True
    audit = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert audit["unknown_or_orphan_files"] == []


def test_interrupted_finalize_lock_history_is_recovered_and_classified(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "interrupted_finalize_history"
    history = output / "_lock_history" / "finalize_interrupted"
    history.mkdir(parents=True)
    subject._atomic_json(
        {
            "schema_version": "t9.1.3-finalize-lock-v1",
            "task_id": subject.TASK_ID,
            "pid": 2_147_483_647,
            "hostname": socket.gethostname(),
            "token": "dead-finalizer",
        },
        history / "owner.json",
    )
    (history / ".recovery.json.host.999.1.tmp").write_bytes(b"partial-recovery")
    subject._recover_incomplete_lock_history(output)
    recovery = json.loads((history / "recovery.json").read_text(encoding="utf-8"))
    assert len(recovery["archived_files"]) == 2
    audit = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert audit["unknown_or_orphan_files"] == []


def test_finalization_atomic_temp_is_retained_inside_the_manifested_tree(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "finalization_atomic"
    target = tmp_path / "published-ledger.parquet"
    temporary = target.parent / f".{target.name}.host.999.1.tmp"
    temporary.write_bytes(b"partial-parquet-footer")
    recovery_path = subject._recover_finalization_atomic_temporaries(
        output, [target]
    )
    assert recovery_path is not None and recovery_path.is_file()
    assert not temporary.exists()
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert recovery["original_paths"] == [
        temporary.resolve().relative_to(ROOT.resolve()).as_posix()
    ]
    archived = Path(recovery["archived_files"][0]["path"])
    assert archived.read_bytes() == b"partial-parquet-footer"
    audit = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    assert audit["unknown_or_orphan_files"] == []


def test_atomic_json_uses_unique_temporary_names_under_concurrent_writers(
    tmp_path: Path,
) -> None:
    target = tmp_path / "atomic.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda value: subject._atomic_json({"value": value}, target),
                range(64),
            )
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["value"] in range(64)
    assert not list(tmp_path.glob(".*.tmp"))


def test_finalize_namespace_lock_excludes_agent_writers_and_releases(
    tmp_path: Path,
) -> None:
    output = tmp_path / "finalize_lock"
    with subject._namespace_finalize_lock(output):
        with pytest.raises(
            subject.ConcurrentAgentWriterError, match="finalization owns"
        ):
            subject._assert_finalize_not_active(output)
        with pytest.raises(
            subject.ConcurrentAgentWriterError, match="live finalizer"
        ):
            with subject._namespace_finalize_lock(output):
                raise AssertionError("nested finalizer should never enter")
    subject._assert_finalize_not_active(output)
    assert not (output / "_locks" / "finalize.lock").exists()


def test_orphan_attempt_cannot_borrow_an_existing_canonical_success(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "orphan"
    subject.train_population(
        config, output_dir=output, family="mf", production=False
    )
    seed = int(config["training"]["paired_root_seeds"][0])
    orphan = output / "_training_attempts" / f"mf_seed_{seed}_orphan"
    orphan.mkdir()
    subject._recover_interrupted_training_attempts(
        output_dir=output,
        strategy="mf",
        seed=seed,
        config=config,
        directional=subject._strategy_config(config, "mf"),
        implementation_hash=subject.implementation_sha256(),
    )
    assert not (orphan / "outcome.json").exists()
    failure = json.loads((orphan / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_subtype"] == "START_MARKER_INTERRUPTED"
    assert failure["attempt_id"] == orphan.name


def test_interrupted_outcome_is_recovered_only_for_matching_attempt_id(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "outcome_recovery"
    subject.train_population(
        config, output_dir=output, family="mf", production=False
    )
    seed = int(config["training"]["paired_root_seeds"][0])
    attempts = list((output / "_training_attempts").glob(f"mf_seed_{seed}_*"))
    assert len(attempts) == 1
    attempt = attempts[0]
    (attempt / "outcome.json").unlink()
    subject._recover_interrupted_training_attempts(
        output_dir=output,
        strategy="mf",
        seed=seed,
        config=config,
        directional=subject._strategy_config(config, "mf"),
        implementation_hash=subject.implementation_sha256(),
    )
    outcome = json.loads((attempt / "outcome.json").read_text(encoding="utf-8"))
    record = json.loads(
        subject._agent_paths(output, "mf", seed)[2].read_text(encoding="utf-8")
    )
    assert outcome["attempt_id"] == record["attempt_id"] == attempt.name
    assert outcome["recovered_after_interrupted_outcome_write"] is True


def test_corrupt_success_is_superseded_before_retraining_and_remains_auditable(
    tmp_path: Path,
) -> None:
    config = _pilot_config()
    output = tmp_path / "superseded"
    subject.train_population(
        config, output_dir=output, family="all", production=False
    )
    seed = int(config["training"]["paired_root_seeds"][0])
    weight_path, _, record_path = subject._agent_paths(output, "mf", seed)
    old_record = json.loads(record_path.read_text(encoding="utf-8"))
    weight_path.write_bytes(b"deliberately-corrupt-success")
    retrained = subject.train_population(
        config, output_dir=output, family="mf", production=False
    )
    assert retrained["newly_trained_agents"] == 1
    new_record = json.loads(record_path.read_text(encoding="utf-8"))
    assert new_record["attempt_id"] != old_record["attempt_id"]
    audit = subject._audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=subject.implementation_sha256(),
    )
    states = Counter(row["state"] for row in audit["training_attempts"])
    assert states["SUPERSEDED_CORRUPT"] == 1
    assert states["SUCCESS"] == 2
    assert audit["all_training_attempts_terminal_and_valid"] is True
    assert audit["every_agent_has_successful_attempt"] is True
    assert audit["unknown_or_orphan_files"] == []


def test_six_state_adapter_retains_full_three_method_two_stratum_census() -> None:
    pytest.importorskip("torch")
    config = _pilot_config()
    root_seed = int(config["training"]["paired_root_seeds"][0])
    selected = {
        strategy: (
            subject.build_policy(
                strategy, subject._strategy_config(config, strategy), root_seed
            ),
            {"root_seed": root_seed},
        )
        for strategy in ("mf", "nmf")
    }
    trajectories, events, reconstruction = subject._evaluate_six_state(
        config, selected
    )

    assert len(trajectories) == 6 * 3 * 2
    assert len(events) == len(trajectories) * 4
    assert {row["initial_state"] for row in trajectories} == set(STATE_LABELS)
    assert {row["strategy"] for row in trajectories} == {"standard", "mf", "nmf"}
    assert len({row["stratum"] for row in trajectories}) == 2
    census = Counter(
        (row["stratum"], row["strategy"], row["initial_state"])
        for row in trajectories
    )
    assert set(census.values()) == {1}
    assert all(
        row["postselected"] is False
        and row["included_in_denominator"] is True
        and row["status"] == "COMPLETE"
        for row in trajectories + events
    )
    assert all(
        row["code_weighted_I"] == pytest.approx(row["code_survival"], abs=1.0e-10)
        for row in trajectories
    )
    assert all(
        2.0 * row["state_fidelity_unconditional"]
        == pytest.approx(
            row["code_survival"] + row["signed_axis_signal"], abs=1.0e-9
        )
        for row in trajectories
    )
    assert {
        (row["stratum"], row["cutoff"]) for row in trajectories
    } == {("primary_cutoff4", 4), ("selected_confirmation_cutoff4", 4)}
    assert all(
        np.isfinite(row["selected_branch_probability"])
        and 0.0 < row["selected_branch_probability"] <= 1.0
        for row in events
    )
    for row in events:
        assert row["layer_control_decision_prefix_length"] == row["half_cycle"] - 1
        assert row["physical_layer_application_half_cycle"] == row["half_cycle"]
        assert row["measurement_outcome_half_cycle"] == row["half_cycle"]
        assert (
            row["post_measurement_control_decision_prefix_length"]
            == row["half_cycle"]
        )
        assert (
            row["post_measurement_virtual_rotation_after_half_cycle"]
            == row["half_cycle"]
        )
        terminal = row["half_cycle"] == 4
        assert row["post_measurement_control_layer_application_half_cycle"] == (
            None if terminal else row["half_cycle"] + 1
        )
        assert row["uses_current_measurement_outcome_for_post_measurement_control"] is True
        assert row["terminal_feedback_correction"] is terminal
        project = np.asarray(json.loads(row["project_action_order_json"]))
        public_gqf = np.asarray(json.loads(row["gqf_action_order_json"]))
        post = np.asarray(
            json.loads(row["post_measurement_project_action_order_json"])
        )
        post_public_gqf = np.asarray(
            json.loads(row["post_measurement_gqf_action_order_json"])
        )
        assert project.shape == public_gqf.shape == post.shape == post_public_gqf.shape == (15,)
        assert np.array_equal(subject.gqf_to_project_action(public_gqf), project)
        assert np.array_equal(subject.gqf_to_project_action(post_public_gqf), post)
    assert subject._event_timeline_valid(events, trajectories, half_cycles=4)
    assert len(reconstruction) == 2
    assert all(set(value) == {"standard", "mf", "nmf"} for value in reconstruction.values())
    assert all(
        method["paper_lifetime_value"] is None
        and method["paper_lifetime_state"] == subject.PAPER_NUMERIC_STATE
        and "empirical_linear_inversion_subchannel_diagnostic" in method
        and method["maximum_joint_trace_error"] <= 2.0e-9
        and method["maximum_joint_hermiticity_error"] <= 2.0e-9
        and method["minimum_final_joint_eigenvalue"] >= -2.0e-8
        for stratum in reconstruction.values()
        for method in stratum.values()
    )


@pytest.mark.parametrize(
    "mutation",
    ("prefix", "terminal", "pending_action", "cluster"),
)
def test_event_timeline_rejects_causal_and_pairing_mutations(
    mutation: str,
) -> None:
    pytest.importorskip("torch")
    config = _pilot_config()
    seed = int(config["training"]["paired_root_seeds"][0])
    selected = {
        strategy: (
            subject.build_policy(
                strategy, subject._strategy_config(config, strategy), seed
            ),
            {"root_seed": seed},
        )
        for strategy in ("mf", "nmf")
    }
    trajectories, events, _ = subject._evaluate_six_state(config, selected)
    changed = deepcopy(events)
    if mutation == "prefix":
        changed[0]["post_measurement_control_decision_prefix_length"] += 1
    elif mutation == "terminal":
        terminal = next(row for row in changed if row["terminal_feedback_correction"])
        terminal["terminal_feedback_correction"] = False
    elif mutation == "pending_action":
        first = changed[0]
        value = np.asarray(
            json.loads(first["post_measurement_project_action_order_json"]),
            dtype=np.float64,
        )
        value[0] += 1.0
        first["post_measurement_project_action_order_json"] = json.dumps(
            value.tolist(), separators=(",", ":")
        )
        first["post_measurement_gqf_action_order_json"] = json.dumps(
            subject.project_to_gqf_action(value).tolist(), separators=(",", ":")
        )
        first["post_measurement_project_action_sha256"] = hashlib.sha256(
            np.ascontiguousarray(value).tobytes()
        ).hexdigest()
    elif mutation == "cluster":
        changed[0]["evaluation_cluster_id"] = "forged-cluster"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    assert not subject._event_timeline_valid(
        changed, trajectories, half_cycles=4
    )


def test_validate_report_rejects_stale_analysis_hash() -> None:
    report = _minimal_valid_report()
    subject.validate_report(
        report, verify_live_files=False, _structural_test_only=True
    )
    report["status"] = "MUTATED_AFTER_SEAL"
    with pytest.raises(ValueError, match="analysis hash mismatch"):
        subject.validate_report(
            report, verify_live_files=False, _structural_test_only=True
        )


def test_validate_report_rejects_resealed_deviation_summary_tamper() -> None:
    report = _minimal_valid_report()
    report["deviation_ledger"]["sha256"] = "f" * 64
    _reseal(report)
    with pytest.raises(ValueError, match="deviation-ledger summary drifted"):
        subject.validate_report(
            report, verify_live_files=False, _structural_test_only=True
        )


def test_nonlive_validator_is_not_a_public_pass_channel() -> None:
    with pytest.raises(ValueError, match="internal structural-test scope"):
        subject.validate_report(_minimal_valid_report(), verify_live_files=False)


def test_mutation_audit_reseals_and_reaches_semantic_checks() -> None:
    audit = subject._mutation_audit(_minimal_valid_report())
    assert audit["all_detected"] is True
    expected = (
        len(subject.REQUIRED_GATE_NAMES)
        + len(subject.SOURCE_EVIDENCE_MUTATION_IDS)
        + len(subject.CLAIM_MUTATION_IDS)
    )
    assert audit["detected_count"] == audit["mutation_count"] == expected
    ids = tuple(row["mutation_id"] for row in audit["mutations"])
    assert all(
        f"published_gate_false:{name}" in ids
        for name in subject.REQUIRED_GATE_NAMES
    )
    assert all(name in ids for name in subject.SOURCE_EVIDENCE_MUTATION_IDS)
    assert not any(name.startswith("gate_false:") for name in ids)
    assert all(
        row["detection_reason"]
        and "analysis hash mismatch" not in row["detection_reason"]
        for row in audit["mutations"]
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("evidence_grade", "evidence grade escalation"),
        ("external_claim", "claim slot was illegally populated"),
        ("numeric", "paper-scale numerical boundary drifted"),
        ("gate", "required T9.1.3 gates failed"),
        ("manifest", "artifact manifest root mismatch"),
        ("unknown_top", "report schema drifted"),
        ("unknown_nested", "six_state_evaluator schema drifted"),
        ("action_ranking", "action contract drifted"),
        ("protocol", "protocol transcription drifted"),
        ("qualification_scope", "six-state report semantics"),
        ("selection_description", "selection contract drifted"),
        ("training_runtime", "training/finalization runtime contract drifted"),
        ("raw_replay", "six-state report semantics"),
        ("failure_runtime", "failure-attempt report contract drifted"),
    ),
)
def test_validate_report_rejects_resealed_claim_gate_and_manifest_tampering(
    mutation: str, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _minimal_valid_report()
    if mutation == "manifest":
        runtime = deepcopy(report["runtime_profile"]["signature"])
        runtime_hash = subject._canonical_sha256(runtime)
        monkeypatch.setattr(
            subject,
            "_current_training_runtime",
            lambda: (deepcopy(runtime), runtime_hash),
        )
    if mutation == "evidence_grade":
        report["evidence_grade"] = "OFFICIAL_EXACT_REPRODUCTION"
    elif mutation == "external_claim":
        report["external_claim_slots"]["puviani_surpass"] = {"gain": 1.2}
    elif mutation == "numeric":
        report["paper_scale_numerical_result"]["value"] = 1500.0
    elif mutation == "gate":
        report["gates"]["identity_and_schema_frozen"] = False
    elif mutation == "manifest":
        report["artifact_manifest"]["sorted_manifest_sha256"] = "0" * 64
    elif mutation == "unknown_top":
        report["claimed_paper_T_ch_cycles"] = 1500.0
    elif mutation == "unknown_nested":
        report["six_state_evaluator"]["paper_lifetime_sota_cycles"] = 1500.0
    elif mutation == "action_ranking":
        report["action_contract"]["ranking_eligibility"] = "ELIGIBLE_SOTA"
    elif mutation == "protocol":
        report["protocol_transcription"]["selection"] = "paper_test_best_of_20"
    elif mutation == "qualification_scope":
        report["six_state_evaluator"]["qualification_scope"] = "paper lifetime SOTA"
    elif mutation == "selection_description":
        report["selection_contract"]["within_agent"] = "best test checkpoint"
    elif mutation == "training_runtime":
        report["training_runtime_contract"]["signature"][
            "python_version"
        ] += "-tampered"
    elif mutation == "raw_replay":
        report["six_state_evaluator"]["raw_replay_audit"][
            "maximum_projected_rho_absolute_error"
        ] = 1.0e-3
    elif mutation == "failure_runtime":
        report["failure_attempt_ledger"][
            "all_failure_runtime_signatures_self_bound"
        ] = False
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)
    _reseal(report)
    with pytest.raises(ValueError, match=match):
        subject.validate_report(
            report,
            verify_live_files=mutation == "manifest",
            _structural_test_only=True,
        )


def test_production_artifact_live_validation_when_available() -> None:
    if not PRODUCTION_REPORT.is_file():
        pytest.skip("T9.1.3 production artifact has not been generated yet")
    report = subject.validate_report(PRODUCTION_REPORT, verify_live_files=True)
    assert report["population"]["total_agent_count"] == 40
    assert report["population"]["agents_per_family"] == {"mf": 20, "nmf": 20}
    assert report["training_ledger"]["rows"] == 40_000
    evaluator = report["six_state_evaluator"]
    assert tuple(evaluator["state_labels"]) == STATE_LABELS
    assert evaluator["methods"] == ["standard", "mf", "nmf"]
    assert evaluator["trajectory_rows"] == 1008
    assert evaluator["event_rows"] == 20_160
    assert evaluator["postselection"] is False


def test_implementation_hash_binds_executed_parent_protocol_verifier() -> None:
    assert (
        "cnn_fpga/benchmark/phase9_three_lane_protocol.py"
        in subject.IMPLEMENTATION_RELATIVE_PATHS
    )
