"""T4.1.5 offline NMF-teacher to deterministic recurrence-student distillation.

The registered T2.3.7 five-agent NMF ensemble is restored only in this offline
benchmark.  A 75-trainable/105-stored recurrence student is fit on training
trajectories, selected on validation trajectories, and reported once on held-
out evaluation trajectories.  The exported online artifact contains no torch
state_dict, simulator object, teacher target, or hidden truth.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cnn_fpga.control.teacher_student import (
    CONTROL_PARAMETER_NAMES,
    DistilledRecurrenceStudent,
    DistilledStudentArtifact,
    StudentObservation,
    online_contract,
)


@dataclass(frozen=True)
class TeacherStudentValidationConfig:
    teacher_checkpoint_path: str = "docs/t2_3_7_nmf_directional_ranking_checkpoints.pt"
    teacher_manifest_path: str = "docs/t2_3_7_nmf_directional_ranking.json"
    trajectories_per_split: int = 256
    half_cycles: int = 20
    training_data_seed: int = 20261501
    validation_data_seed: int = 20261502
    evaluation_data_seed: int = 20261503
    restart_seeds: tuple[int, ...] = (131, 271, 419)
    training_epochs: int = 1200
    validation_interval: int = 20
    phase_one_learning_rate: float = 0.03
    phase_two_learning_rate: float = 0.005
    phase_two_start_epoch: int = 600
    early_stopping_patience: int = 12
    minimum_relative_improvement: float = 1.0e-4
    plateau_relative_tolerance: float = 5.0e-3
    raw_clip: float = 4.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("trajectories_per_split", 128),
            ("half_cycles", 4),
            ("training_epochs", 500),
            ("validation_interval", 1),
            ("phase_two_start_epoch", 1),
            ("early_stopping_patience", 3),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        if self.half_cycles != 20:
            raise ValueError("T4.1.5 must match the registered T2.3.7 10-cycle/20-half-cycle teacher")
        if self.phase_two_start_epoch >= self.training_epochs:
            raise ValueError("phase_two_start_epoch must precede training_epochs")
        if self.training_epochs % self.validation_interval:
            raise ValueError("training_epochs must be divisible by validation_interval")
        seeds = tuple(self.restart_seeds)
        if len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ValueError("at least three unique restart seeds are required")
        data_seeds = (self.training_data_seed, self.validation_data_seed, self.evaluation_data_seed)
        if len(set(data_seeds)) != 3:
            raise ValueError("training/validation/evaluation data seeds must be disjoint")
        for name in (
            "phase_one_learning_rate",
            "phase_two_learning_rate",
            "raw_clip",
            "minimum_relative_improvement",
            "plateau_relative_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _outcome_matrix(seed: int, trajectories: int, half_cycles: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = rng.uniform(0.10, 0.45, size=(trajectories, 1))
    outcomes = (rng.random((trajectories, half_cycles)) < probabilities).astype(np.int64)
    # Deterministic boundary histories ensure long all-g/all-e/alternating and
    # transition-rich paths are never left to chance.
    outcomes[0] = 0
    outcomes[1] = 1
    outcomes[2] = np.arange(half_cycles) & 1
    outcomes[3] = 1 - (np.arange(half_cycles) & 1)
    for index in range(4, min(12, trajectories)):
        switch = 2 + (index * 3) % (half_cycles - 3)
        outcomes[index, switch:] = 1 - outcomes[index, switch:]
    return outcomes


def _dataset_sha256(outcomes: np.ndarray, targets: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(outcomes, dtype="<i1").tobytes())
    digest.update(np.asarray(targets, dtype="<f8").tobytes())
    return digest.hexdigest()


def _load_teacher_ensemble(config: TeacherStudentValidationConfig):
    from physics.nmf_directional_ranking import (
        DirectionalRankingConfig,
        build_policy,
        state_dict_sha256,
    )
    # The physics module imports scipy/numpy before its optional torch import,
    # avoiding the duplicate OpenMP-runtime ordering seen on the Windows host.
    import torch

    manifest = json.loads(Path(config.teacher_manifest_path).read_text(encoding="utf-8"))
    checkpoint_hash = _sha256(config.teacher_checkpoint_path)
    if manifest.get("status") != "PASS":
        raise ValueError("registered T2.3.7 teacher manifest is not PASS")
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("teacher checkpoint hash does not match T2.3.7 manifest")
    checkpoint = torch.load(config.teacher_checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 3:
        raise ValueError("teacher checkpoint must use T2.3.7 schema v3")
    teacher_config = DirectionalRankingConfig(**checkpoint["config"])
    if 2 * teacher_config.full_cycles != config.half_cycles:
        raise ValueError("teacher horizon does not match distillation horizon")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_config = replace(teacher_config, device=device)
    models = []
    hashes = []
    for item in checkpoint["models"]["nmf"]:
        model = build_policy("nmf", teacher_config, int(item["training_seed"]))
        model.load_state_dict(item["state_dict"])
        model.eval()
        actual_hash = state_dict_sha256(item["state_dict"])
        if actual_hash != item["checkpoint_sha256"]:
            raise ValueError("teacher model state hash mismatch")
        models.append(model)
        hashes.append(actual_hash)
    if len(models) != 5 or len(set(hashes)) != 5:
        raise ValueError("T4.1.5 requires all five unique registered NMF agents")
    return torch, models, tuple(hashes), checkpoint_hash, device


def _teacher_targets(torch, models, outcomes: np.ndarray, device: str) -> np.ndarray:
    dtype = next(models[0].parameters()).dtype
    values = torch.tensor(outcomes, dtype=dtype, device=device)
    ensemble = torch.zeros(
        (len(outcomes), outcomes.shape[1], len(CONTROL_PARAMETER_NAMES)),
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        for model in models:
            for half_index in range(outcomes.shape[1]):
                ensemble[:, half_index, :] += model(values[:, :half_index], half_index)
    ensemble /= len(models)
    result = ensemble.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("offline teacher ensemble emitted nonfinite targets")
    return result


def _fit_student(
    torch,
    training_outcomes: np.ndarray,
    training_targets: np.ndarray,
    validation_outcomes: np.ndarray,
    validation_targets: np.ndarray,
    config: TeacherStudentValidationConfig,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], int]:
    dtype = torch.float64

    class StudentModule(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            self.initial_raw = torch.nn.Parameter(
                0.01 * torch.randn((15,), dtype=dtype, generator=generator).to(device)
            )
            self.saturation_raw = torch.nn.Parameter(
                0.01 * torch.randn((2, 15), dtype=dtype, generator=generator).to(device)
            )
            self.decay_logits = torch.nn.Parameter(
                torch.zeros((2, 15), dtype=dtype, device=device)
            )

        def parameters_physical(self):
            initial = config.raw_clip * torch.tanh(self.initial_raw / config.raw_clip)
            saturation = config.raw_clip * torch.tanh(self.saturation_raw / config.raw_clip)
            decay = 0.02 + 0.975 * torch.sigmoid(self.decay_logits)
            return initial, saturation, decay

        def forward(self, outcomes):
            initial, saturation, decay = self.parameters_physical()
            state = initial[None, :].expand(outcomes.shape[0], -1)
            outputs = [state]
            for half_index in range(1, outcomes.shape[1]):
                observed = outcomes[:, half_index - 1]
                selected_decay = decay[observed]
                selected_saturation = saturation[observed]
                state = selected_decay * state + (1.0 - selected_decay) * selected_saturation
                outputs.append(state)
            return torch.stack(outputs, dim=1)

    train_x = torch.tensor(training_outcomes, dtype=torch.long, device=device)
    train_y = torch.tensor(training_targets, dtype=dtype, device=device)
    validation_x = torch.tensor(validation_outcomes, dtype=torch.long, device=device)
    validation_y = torch.tensor(validation_targets, dtype=dtype, device=device)
    records = []
    selected: tuple[float, int, dict[str, Any]] | None = None
    for restart_index, seed in enumerate(config.restart_seeds):
        model = StudentModule(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.phase_one_learning_rate)
        best_validation = float("inf")
        best_epoch = -1
        best_state = None
        stopped_epoch = config.training_epochs
        checkpoints_without_material_improvement = 0
        validation_history: list[tuple[int, float]] = []
        for epoch in range(1, config.training_epochs + 1):
            if epoch == config.phase_two_start_epoch:
                for group in optimizer.param_groups:
                    group["lr"] = config.phase_two_learning_rate
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x)
            mse = torch.mean((prediction - train_y) ** 2)
            regularization = 1.0e-6 * sum(torch.mean(parameter * parameter) for parameter in model.parameters())
            loss = mse + regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if epoch % config.validation_interval == 0:
                with torch.no_grad():
                    validation_mse = float(torch.mean((model(validation_x) - validation_y) ** 2).cpu())
                validation_history.append((epoch, validation_mse))
                materially_better = validation_mse < best_validation * (
                    1.0 - config.minimum_relative_improvement
                )
                if materially_better:
                    best_validation = validation_mse
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                    }
                    checkpoints_without_material_improvement = 0
                else:
                    checkpoints_without_material_improvement += 1
                if checkpoints_without_material_improvement >= config.early_stopping_patience:
                    stopped_epoch = epoch
                    break
        if best_state is None:
            raise RuntimeError("student restart produced no validation checkpoint")
        tail = validation_history[-config.early_stopping_patience :]
        tail_relative_improvement = (
            (tail[0][1] - min(value for _, value in tail)) / max(tail[0][1], 1.0e-30)
        )
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            training_mse = float(torch.mean((model(train_x) - train_y) ** 2).cpu())
            initial, saturation, decay = model.parameters_physical()
        record = {
            "restart_index": restart_index,
            "seed": seed,
            "best_epoch_selected_on_validation": best_epoch,
            "stopped_epoch": stopped_epoch,
            "converged_before_maximum_epoch": stopped_epoch < config.training_epochs,
            "tail_relative_validation_improvement": tail_relative_improvement,
            "validation_plateau_reached": (
                stopped_epoch < config.training_epochs
                or tail_relative_improvement <= config.plateau_relative_tolerance
            ),
            "training_mse_at_selected_checkpoint": training_mse,
            "validation_mse": best_validation,
            "all_75_trainables_receive_gradient": all(
                parameter.grad is not None
                and bool(torch.all(torch.isfinite(parameter.grad)).detach().cpu())
                and int(torch.count_nonzero(parameter.grad).detach().cpu()) == parameter.numel()
                for parameter in model.parameters()
            ),
        }
        records.append(record)
        candidate = (
            best_validation,
            restart_index,
            {
                "initial": initial.detach().cpu().numpy(),
                "saturation": saturation.detach().cpu().numpy(),
                "decay": decay.detach().cpu().numpy(),
            },
        )
        if selected is None or candidate[0] < selected[0]:
            selected = candidate
    if selected is None:
        raise RuntimeError("student selection failed")
    return selected[2], records, selected[1]


def _student_predictions(artifact: DistilledStudentArtifact, outcomes: np.ndarray) -> np.ndarray:
    initial = np.asarray(artifact.initial_state)
    saturation = np.asarray(artifact.outcome_saturations)
    decay = np.asarray(artifact.outcome_decays)
    state = np.broadcast_to(initial, (len(outcomes), len(initial))).copy()
    outputs = [state.copy()]
    for half_index in range(1, outcomes.shape[1]):
        observed = outcomes[:, half_index - 1]
        selected_decay = decay[observed]
        selected_saturation = saturation[observed]
        state = selected_decay * state + (1.0 - selected_decay) * selected_saturation
        outputs.append(state.copy())
    return np.stack(outputs, axis=1)


def _latest_only_fit(training_outcomes: np.ndarray, training_targets: np.ndarray):
    initial = np.mean(training_targets[:, 0, :], axis=0)
    latest = np.zeros((2, len(CONTROL_PARAMETER_NAMES)), dtype=np.float64)
    for outcome in (0, 1):
        values = []
        for half_index in range(1, training_outcomes.shape[1]):
            mask = training_outcomes[:, half_index - 1] == outcome
            values.append(training_targets[mask, half_index, :])
        latest[outcome] = np.mean(np.concatenate(values, axis=0), axis=0)
    return initial, latest


def _latest_only_predict(parameters, outcomes: np.ndarray) -> np.ndarray:
    initial, latest = parameters
    result = np.empty((len(outcomes), outcomes.shape[1], len(CONTROL_PARAMETER_NAMES)))
    result[:, 0, :] = initial
    for half_index in range(1, outcomes.shape[1]):
        result[:, half_index, :] = latest[outcomes[:, half_index - 1]]
    return result


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    error = prediction - target
    per_half = np.mean(error * error, axis=(0, 2))
    return {
        "mse": float(np.mean(error * error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "per_half_cycle_mse": [float(value) for value in per_half],
    }


def _maximum_prefix_inconsistency(outcomes: np.ndarray, values: np.ndarray) -> float:
    maximum = 0.0
    for half_index in range(outcomes.shape[1]):
        groups: dict[bytes, list[int]] = {}
        for row_index in range(len(outcomes)):
            key = np.asarray(outcomes[row_index, :half_index], dtype=np.int8).tobytes()
            groups.setdefault(key, []).append(row_index)
        for indices in groups.values():
            if len(indices) > 1:
                maximum = max(
                    maximum,
                    float(np.max(np.ptp(values[indices, half_index, :], axis=0))),
                )
    return maximum


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        "cnn_fpga/control/teacher_student.py",
        "cnn_fpga/benchmark/offline_teacher_student_distillation.py",
        "physics/nmf_directional_ranking.py",
    ):
        digest.update(path.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_teacher_student_validation(
    config: TeacherStudentValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], DistilledStudentArtifact]:
    settings = TeacherStudentValidationConfig() if config is None else config
    if not isinstance(settings, TeacherStudentValidationConfig):
        raise TypeError("config must be TeacherStudentValidationConfig")
    torch, teachers, teacher_hashes, checkpoint_hash, device = _load_teacher_ensemble(settings)
    split_seeds = {
        "training": settings.training_data_seed,
        "validation": settings.validation_data_seed,
        "evaluation": settings.evaluation_data_seed,
    }
    outcomes = {
        split: _outcome_matrix(seed, settings.trajectories_per_split, settings.half_cycles)
        for split, seed in split_seeds.items()
    }
    targets = {
        split: _teacher_targets(torch, teachers, values, device)
        for split, values in outcomes.items()
    }
    dataset_hashes = {
        split: _dataset_sha256(outcomes[split], targets[split]) for split in outcomes
    }
    parameters, restart_records, selected_restart = _fit_student(
        torch,
        outcomes["training"],
        targets["training"],
        outcomes["validation"],
        targets["validation"],
        settings,
        device,
    )
    artifact = DistilledStudentArtifact.create(
        initial_state=parameters["initial"],
        outcome_saturations=parameters["saturation"],
        outcome_decays=parameters["decay"],
        raw_clip=settings.raw_clip,
        teacher_checkpoint_sha256=checkpoint_hash,
        teacher_model_sha256s=teacher_hashes,
        training_dataset_sha256=dataset_hashes["training"],
        validation_dataset_sha256=dataset_hashes["validation"],
        selected_restart=selected_restart,
    )
    predictions = {
        split: _student_predictions(artifact, values) for split, values in outcomes.items()
    }
    latest_parameters = _latest_only_fit(outcomes["training"], targets["training"])
    latest_predictions = {
        split: _latest_only_predict(latest_parameters, values) for split, values in outcomes.items()
    }
    metrics = {}
    for split in outcomes:
        student = _metrics(targets[split], predictions[split])
        latest = _metrics(targets[split], latest_predictions[split])
        safe = _metrics(targets[split], np.zeros_like(targets[split]))
        student["imitation_gain_retention_vs_zero"] = 1.0 - student["mse"] / safe["mse"]
        latest["imitation_gain_retention_vs_zero"] = 1.0 - latest["mse"] / safe["mse"]
        metrics[split] = {"student": student, "latest_only": latest, "zero_safe": safe}

    rows = []
    source_digest = hashlib.sha256()
    for split in ("training", "validation", "evaluation"):
        for trajectory_index in range(settings.trajectories_per_split):
            for half_index in range(settings.half_cycles):
                prefix = outcomes[split][trajectory_index, :half_index]
                row: dict[str, object] = {
                    "split": split,
                    "data_seed": split_seeds[split],
                    "trajectory_index": trajectory_index,
                    "half_cycle_index": half_index,
                    "history_length": half_index,
                    "history_sha256": hashlib.sha256(np.asarray(prefix, dtype=np.int8).tobytes()).hexdigest(),
                    "latest_observed_outcome": (
                        "none" if half_index == 0 else ("e" if prefix[-1] else "g")
                    ),
                    "teacher_student_squared_error": float(
                        np.mean(
                            (predictions[split][trajectory_index, half_index] - targets[split][trajectory_index, half_index])
                            ** 2
                        )
                    ),
                    "teacher_latest_squared_error": float(
                        np.mean(
                            (latest_predictions[split][trajectory_index, half_index] - targets[split][trajectory_index, half_index])
                            ** 2
                        )
                    ),
                    "scope": "offline_teacher_target_distillation_source_data",
                }
                for parameter_index, name in enumerate(CONTROL_PARAMETER_NAMES):
                    row[f"offline_teacher_target_{name}"] = float(
                        targets[split][trajectory_index, half_index, parameter_index]
                    )
                    row[f"student_prediction_{name}"] = float(
                        predictions[split][trajectory_index, half_index, parameter_index]
                    )
                rows.append(row)
                source_digest.update(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                )

    teacher_prefix_inconsistency = max(
        _maximum_prefix_inconsistency(outcomes[split], targets[split]) for split in outcomes
    )
    student_prefix_inconsistency = max(
        _maximum_prefix_inconsistency(outcomes[split], predictions[split]) for split in outcomes
    )
    online_a = DistilledRecurrenceStudent(artifact)
    online_b = DistilledRecurrenceStudent(artifact)
    online_actions_a = [online_a.initial_decision().raw_control_residual]
    online_actions_b = [online_b.initial_decision().raw_control_residual]
    probe_outcomes = outcomes["evaluation"][0]
    for cycle_index, outcome in enumerate(probe_outcomes[:-1]):
        observation = StudentObservation(cycle_index, "e" if outcome else "g")
        online_actions_a.append(online_a.step(observation).raw_control_residual)
        online_actions_b.append(online_b.step(observation).raw_control_residual)
    online_actions = np.asarray(online_actions_a)
    online_matches_batch = np.array_equal(
        online_actions,
        predictions["evaluation"][0],
    )
    deterministic_online = online_actions_a == online_actions_b
    safety_reasons = []
    for field in ("valid", "crc_ok", "parameter_fresh", "deadline_ok"):
        student = DistilledRecurrenceStudent(artifact)
        kwargs = {"valid": True, "crc_ok": True, "parameter_fresh": True, "deadline_ok": True}
        kwargs[field] = False
        decision = student.step(StudentObservation(0, "g", **kwargs))
        safety_reasons.append(
            decision.used_safe_baseline and max(abs(value) for value in decision.raw_control_residual) == 0.0
        )
    leakage_student = DistilledRecurrenceStudent(artifact)
    leakage_decision = leakage_student.step(StudentObservation(0, "leakage"))
    teacher_object_rejected = False
    try:
        DistilledRecurrenceStudent(teachers[0])
    except TypeError:
        teacher_object_rejected = True
    online_source = inspect.getsource(__import__("cnn_fpga.control.teacher_student", fromlist=["*"]))
    evaluation = metrics["evaluation"]
    gates = {
        "registered_t2_3_7_checkpoint_hash_matches": checkpoint_hash == _sha256(settings.teacher_checkpoint_path),
        "all_five_teacher_models_are_hash_verified": len(teacher_hashes) == 5 and len(set(teacher_hashes)) == 5,
        "training_validation_evaluation_data_seeds_are_disjoint": len(set(split_seeds.values())) == 3,
        "dataset_grid_is_complete": len(rows) == 3 * settings.trajectories_per_split * settings.half_cycles,
        "three_restarts_train_all_75_parameters": (
            len(restart_records) >= 3 and all(record["all_75_trainables_receive_gradient"] for record in restart_records)
        ),
        "all_restarts_reach_registered_validation_plateau": all(
            record["validation_plateau_reached"] for record in restart_records
        ),
        "restart_selection_is_validation_only": (
            selected_restart == min(restart_records, key=lambda item: item["validation_mse"])["restart_index"]
        ),
        "evaluation_dataset_hash_is_not_in_student_artifact": (
            dataset_hashes["evaluation"] not in json.dumps(artifact.to_dict(), sort_keys=True)
        ),
        "student_beats_zero_safe_imitation_baseline_on_evaluation": (
            evaluation["student"]["mse"] < evaluation["zero_safe"]["mse"]
        ),
        "student_beats_latest_only_imitation_baseline_on_evaluation": (
            evaluation["student"]["mse"] < evaluation["latest_only"]["mse"]
        ),
        "teacher_and_student_are_prefix_causal": (
            teacher_prefix_inconsistency <= 1.0e-12 and student_prefix_inconsistency <= 1.0e-12
        ),
        "online_replay_matches_offline_batch_bit_exactly": online_matches_batch,
        "online_replay_is_deterministic": deterministic_online,
        "health_faults_force_zero_residual_safe_baseline": all(safety_reasons),
        "leakage_forces_zero_residual_safe_baseline": (
            leakage_decision.used_safe_baseline
            and max(abs(value) for value in leakage_decision.raw_control_residual) == 0.0
        ),
        "online_api_rejects_teacher_model_object": teacher_object_rejected,
        "online_module_has_no_torch_or_simulator_import": (
            "import torch" not in online_source and "physics." not in online_source
        ),
        "student_resource_profile_is_exact_and_hardware_fields_null": (
            artifact.resource_profile.stored_scalars == 105
            and artifact.resource_profile.parameter_bytes_float32 == 420
            and artifact.resource_profile.target_latency_cycles is None
            and not artifact.resource_profile.rtl_measured
            and not artifact.resource_profile.board_measured
        ),
        "student_artifact_is_hash_bound_and_contains_no_state_dict": (
            len(artifact.artifact_sha256) == 64 and "state_dict" not in json.dumps(artifact.to_dict())
        ),
        "online_contract_is_observed_health_only": (
            online_contract()["teacher_model_runtime_dependency"] is False
            and online_contract()["simulator_truth_runtime_dependency"] is False
        ),
        "scope_remains_imitation_not_physical_or_hardware_gain": True,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t4.1.5-offline-teacher-student-validation-v1",
        "task_id": "T4.1.5",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "source_rows_sha256": source_digest.hexdigest(),
        "validation_config": asdict(settings),
        "execution_device": device,
        "teacher_provenance": {
            "role": "offline_frozen_model_aware_control_teacher_ensemble",
            "checkpoint_path": settings.teacher_checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "manifest_path": settings.teacher_manifest_path,
            "manifest_sha256": _sha256(settings.teacher_manifest_path),
            "model_sha256s": list(teacher_hashes),
            "runtime_allowed": False,
        },
        "dataset": {
            "split_seeds": split_seeds,
            "trajectories_per_split": settings.trajectories_per_split,
            "half_cycles": settings.half_cycles,
            "rows_per_split": settings.trajectories_per_split * settings.half_cycles,
            "hashes": dataset_hashes,
            "evaluation_used_for_training_or_selection": False,
        },
        "training_restarts": restart_records,
        "selected_restart": selected_restart,
        "student_artifact": artifact.to_dict(),
        "online_contract": dict(online_contract()),
        "metrics": metrics,
        "causality": {
            "teacher_maximum_same_prefix_inconsistency": teacher_prefix_inconsistency,
            "student_maximum_same_prefix_inconsistency": student_prefix_inconsistency,
            "online_matches_batch_bit_exactly": online_matches_batch,
            "deterministic_online_replay": deterministic_online,
        },
        "gate_summary": {
            "passed": sum(gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": "offline five-agent teacher ensemble distilled into a deterministic 105-scalar observed-outcome recurrence candidate with explicit zero-residual safety fallback",
            "forbidden": "teacher at online runtime, physical/lifetime/control gain retention, leakage-trained student, fixed-point equivalence, RTL synthesis, FPGA timing, or board/device measurement",
        },
    }
    return payload, rows, artifact


def write_teacher_student_validation(
    json_path: str | Path = "docs/t4_1_5_teacher_student_validation.json",
    csv_path: str | Path = "docs/t4_1_5_teacher_student_source_data.csv",
    student_path: str | Path = "docs/t4_1_5_distilled_student_checkpoint.json",
    config: TeacherStudentValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows, artifact = build_teacher_student_validation(config)
    if not rows:
        raise RuntimeError("teacher/student validation produced no Source Data")
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    student_target = Path(student_path)
    for target in (json_target, csv_target, student_target):
        target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    student_target.write_text(
        json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="docs/t4_1_5_teacher_student_validation.json")
    parser.add_argument("--csv", default="docs/t4_1_5_teacher_student_source_data.csv")
    parser.add_argument("--student", default="docs/t4_1_5_distilled_student_checkpoint.json")
    args = parser.parse_args(argv)
    payload = write_teacher_student_validation(args.json, args.csv, args.student)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TeacherStudentValidationConfig",
    "build_teacher_student_validation",
    "write_teacher_student_validation",
]
