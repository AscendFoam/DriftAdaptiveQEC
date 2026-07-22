"""Auditable paper-constrained MF/NMF artifact lane for Phase 9.

This module deliberately separates two statements which are easy to conflate:

* the project can train and publish a complete, validation-selected population
  of paper-scale MF/NMF controllers; and
* the project has reproduced the paper's cutoff-100, 1000-cycle six-state
  lifetime numbers.

Only the first statement is made by T9.1.3.  The built-in six-state adapter is
an executable finite-cutoff, ten-cycle qualification run with trajectory and
half-cycle raw records.  It refuses to populate a paper-scale lifetime or a
Puviani-surpass field.  The long-sequence dual-backend campaign belongs to
T9.2--T9.6.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.t9_1_3_gpu_attestation import (
    GpuLoadAttestationError,
    gpu_attestation_binding as _gpu_attestation_binding,
    validate_gpu_attestation_binding as _validated_gpu_attestation_binding,
    validate_gpu_load_attestation as _validate_gpu_load_attestation,
)

from physics.differentiable_sbs_trajectory import (
    PARAMETER_NAMES,
    nominal_sbs_parameters,
)
from physics.fock_logical_channel import (
    PAULIS,
    STATE_LABELS,
    logical_eigenstate_density,
    reconstruct_code_subchannel,
)
from physics.nmf_directional_ranking import (
    PAPER_MF_ARCHITECTURE,
    PAPER_RNN_ARCHITECTURE,
    _state_dict_cpu,
    _torch_dtype,
    _training_trajectory_seed,
    state_dict_sha256,
)
from physics.puviani_paper_causal import (
    PAPER_CAUSAL_FEEDBACK_TIMELINE,
    PaperCausalDirectionalRankingConfig as DirectionalRankingConfig,
    PaperCausalSBSConfig,
    PaperCausalSBSTrajectorySimulator,
    applied_control_penalties_per_trajectory,
    build_policy,
    evaluate_policy,
    simulator as _simulator,
)

try:  # The minimal recovery interpreter does not install torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - covered by fail-closed runtime.
    torch = None  # type: ignore[assignment]

try:  # Used only to disambiguate a live PID from a reused PID.
    import psutil
except ModuleNotFoundError:  # pragma: no cover - absence is handled fail closed.
    psutil = None  # type: ignore[assignment]


_ATOMIC_JSON_PROCESS_LOCK = threading.Lock()


# Production training is expected to finish well inside one day, but a laptop
# can be suspended or throttled.  A live owner refreshes its diagnostic lease
# every 15 s.  Same-host recovery also checks the OS process creation time so
# PID reuse cannot masquerade as the original process.  Cross-host locks are
# never stolen automatically (even after the warning horizon): remote liveness
# cannot be proved from a shared directory, so explicit operator recovery is
# required to preserve single-writer safety.
LOCK_HEARTBEAT_SECONDS = 15.0
LOCK_LEASE_SECONDS = 72.0 * 60.0 * 60.0


TASK_ID = "T9.1.3"
SCHEMA_VERSION = "t9.1.3-puviani-paper-constrained-artifacts-v1"
EVIDENCE_GRADE = "PAPER_CONSTRAINED_REIMPLEMENTATION"
STATUS_PASS = "PASS_ARTIFACT_LANE_AND_EXECUTABLE_REIMPLEMENTATION"
SERIAL_RELEASE_SCHEMA_VERSION = "t9.1.3-nmf-after-mf-release-v1"
SERIAL_RELEASE_FILENAME = "nmf_after_mf.release.json"
SERIAL_RELEASE_READY_SCHEMA_VERSION = "t9.1.3-nmf-release-wait-ready-v1"
SERIAL_RELEASE_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "family",
        "prerequisite_family",
        "prerequisite_exit_code",
        "transaction_id",
        "run_dir",
        "attestation_nonce",
        "attestation_sha256",
        "release_nonce",
        "waiter_pid",
        "config_sha256",
        "implementation_sha256",
        "deadline_utc",
        "mf_pid",
        "mf_process_created_unix_ns",
        "released_at_utc",
    }
)
PAPER_NUMERIC_STATE = "INCOMPLETE_NULL"
PAPER_NUMERIC_REASON = (
    "cutoff-100 and 1000-cycle six-state full-denominator evaluation has not "
    "been executed; finite-cutoff ten-cycle qualification is non-promotional"
)
GQF_COMMIT = "c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d"
T9_1_1_ANALYSIS_SHA256 = (
    "c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18"
)
PAPER_DOI = "10.1103/PhysRevLett.134.020601"
PAPER_ARXIV = "2312.07391"
PRODUCTION_CONFIG_SHA256 = (
    "2b708da81a16741b208e47ff3c55da646f7876ee4c743962b18945e66c5e76f0"
)
DEVIATION_LEDGER_SHA256 = (
    "69a75c07452b9dea3e445cf683394663e165a72db47986276cc65dd44078e8e0"
)
GPU_LOAD_ATTESTATION_SCHEMA_VERSION = "t9.1.3-gpu-load-attestation-v1"
GPU_LOAD_ATTESTATION_MAX_AGE_SECONDS = 45.0
GPU_LOAD_ATTESTATION_CLOCK_SKEW_SECONDS = 5.0
GPU_LOAD_ATTESTATION_PURPOSES = frozenset(
    {"TRAINING_LAUNCH", "FINALIZER_LAUNCH"}
)
GPU_LOAD_ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "purpose",
        "config_sha256",
        "implementation_sha256",
        "run_identity",
        "attestation_nonce",
        "sampling_started_at_utc",
        "sampling_completed_at_utc",
        "issued_at_utc",
        "expires_at_utc",
        "target_gpu",
        "load_gate",
        "attestation_sha256",
    }
)
GPU_LOAD_ATTESTATION_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "attestation_sha256",
        "purpose",
        "transaction_id",
        "run_dir",
        "attestation_nonce",
        "target_gpu_uuid",
        "target_gpu_name",
        "target_gpu_total_memory_mib",
        "sampling_completed_at_utc",
        "expires_at_utc",
    }
)
REQUIRED_GATE_NAMES = (
    "identity_and_schema_frozen",
    "parent_protocol_hash_live",
    "paper_constrained_namespace_only",
    "official_commit_pinned",
    "official_live_commit_matches_when_present",
    "paper_bytes_match_preregistered_source",
    "deviation_ledger_is_nontrivial",
    "serious_deviations_revoke_numeric_anchor",
    "twenty_unique_paired_roots",
    "derived_training_and_evaluation_seed_domains_disjoint",
    "both_families_have_twenty_agents",
    "all_forty_training_units_accounted",
    "all_agent_training_runtime_signatures_identical_and_bound",
    "paper_scale_architectures",
    "parameter_counts_match",
    "all_agents_executed_1000_epochs",
    "all_weights_hash_and_reload",
    "all_gradients_and_training_metrics_finite",
    "per_agent_epoch_census_exact",
    "all_training_epochs_lineage_closed_and_endpoints_bound",
    "frozen_family_sentinel_full_retrain_matches_epoch_lineage_and_terminal",
    "validation_candidate_census_complete",
    "all_validation_candidates_replayed_from_checkpoint_and_frozen_seeds",
    "checkpoint_selection_validation_only",
    "one_selected_snapshot_per_agent_and_family",
    "one_selected_agent_per_family",
    "deterministic_family_tie_break",
    "six_states_exact_and_balanced",
    "six_state_semantics_and_paired_clusters_disclosed",
    "standard_mf_nmf_all_qualified",
    "trajectory_census_complete",
    "event_census_complete",
    "raw_schema_matches_frozen_trajectory_and_event_contracts",
    "no_postselection_full_denominator",
    "all_trajectory_statuses_retained",
    "branch_probabilities_valid",
    "event_action_timing_is_causal_and_explicit",
    "six_state_raw_rows_replay_selected_checkpoints_rng_and_physics",
    "code_survival_and_fidelity_finite",
    "six_state_fidelity_pauli_identity",
    "normalized_conditional_joint_state_invariants",
    "empirical_linear_inversion_diagnostics_finite_nonpromotional",
    "action_permutation_roundtrip",
    "action_order_nominal_and_bounds_match_live_backend",
    "all_published_files_manifested_exactly_once",
    "output_tree_has_no_unknown_or_orphan_bytes",
    "training_attempt_ledger_is_terminal_and_complete",
    "all_failure_attempts_are_retained_and_typed",
    "paper_scale_numeric_is_null",
    "official_exact_is_null",
    "puviani_surpass_is_null",
    "native_free_action_lane_not_ranked_phase9",
)
REPORT_CLAIM_BOUNDARY = {
    "allowed": (
        "complete project-owned 20-paired-root MF/NMF artifact population, "
        "validation-only selection, executable six-state raw qualification, "
        "trusted-supervisor GPU-load provenance/self-consistency, and exact "
        "retraining reproducibility for the frozen two-of-forty family sentinels"
    ),
    "forbidden": (
        "official exact reproduction, Puviani surpass, cutoff-100 convergence, "
        "1000-cycle paper lifetime, Phase9 >=1e4-cycle formal lifetime, or "
        "matched Phase9 ranking before the IQ/trusted-codebook adapter; GPU-load "
        "attestations are not cryptographic authenticity/tamper-proof evidence, "
        "and two-of-forty sentinels are not proof against mutation of all 40 agents"
    ),
}
MANDATORY_SHARED_ARTIFACT_BASE_PATHS = (
    "docs/t9_1_3_puviani_agent_registry.csv",
    "docs/t9_1_3_puviani_selection_ledger.csv",
    "docs/t9_1_3_puviani_training_ledger.parquet",
    "docs/t9_1_3_puviani_six_state_trajectories.parquet",
    "docs/t9_1_3_puviani_six_state_events.parquet",
    "configs/phase9/t9_1_3_deviation_ledger.json",
    "configs/phase9/t9_1_3_puviani_paper_constrained.json",
    "docs/t9_1_1_three_lane_protocol.json",
    "docs/t9_1_1_three_lane_protocol_source_data.csv",
    "docs/phase9_three_lane_protocol.md",
    (
        "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
        "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
    ),
)
IMPLEMENTATION_RELATIVE_PATHS = (
    "cnn_fpga/benchmark/puviani_paper_constrained_artifacts.py",
    "cnn_fpga/benchmark/t9_1_3_gpu_attestation.py",
    "cnn_fpga/benchmark/phase9_three_lane_protocol.py",
    "scripts/run_t9_1_3_production.ps1",
    "physics/puviani_paper_causal.py",
    "physics/nmf_directional_ranking.py",
    "physics/differentiable_sbs_trajectory.py",
    "physics/fock_logical_channel.py",
    "physics/fock_density_model.py",
    "physics/sbs_error_space.py",
    "physics/finite_energy_gkp.py",
    "physics/quadrature_conventions.py",
    "physics/constants.py",
    "physics/autonomous_sbs.py",
)
MANDATORY_SHARED_ARTIFACT_PATHS = (
    MANDATORY_SHARED_ARTIFACT_BASE_PATHS
    + IMPLEMENTATION_RELATIVE_PATHS
    + (
        "configs/gqf_official/paper_exact_preregistration.json",
        "configs/gqf_official/patch_manifest.json",
    )
)
CLAIM_MUTATION_IDS = (
    "evidence_grade_escalation",
    "paper_numeric_injection",
    "puviani_surpass_injection",
    "nested_paper_target_injection",
    "nested_external_claim_injection",
    "selection_split_injection",
    "reconstruction_paper_lifetime_injection",
    "namespace_official_exact_injection",
    "published_anchor_eligibility_injection",
    "claim_boundary_deletion",
    "formal_selection_injection",
    "production_contract_downgrade",
    "manifest_census_injection",
    "unknown_top_level_claim_injection",
    "unknown_nested_evaluator_claim_injection",
    "action_ranking_eligibility_injection",
    "protocol_transcription_injection",
    "qualification_scope_injection",
    "selection_description_injection",
)
SOURCE_EVIDENCE_MUTATION_IDS = (
    "source_runtime_signature_tamper",
    "source_raw_replay_witness_tamper",
    "source_output_tree_orphan_injection",
    "source_training_epoch_census_tamper",
    "source_six_state_denominator_tamper",
    "source_failure_runtime_binding_tamper",
    "source_deviation_ledger_hash_tamper",
)

REPORT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "generated_at_utc",
        "status",
        "production_contract",
        "evidence_grade",
        "namespace",
        "t9_1_1_binding",
        "implementation_sha256",
        "config",
        "config_sha256",
        "source_registry",
        "runtime_profile",
        "training_runtime_contract",
        "gpu_load_attestations",
        "protocol_transcription",
        "action_contract",
        "population",
        "selection_contract",
        "six_state_evaluator",
        "training_ledger",
        "failure_attempt_ledger",
        "output_tree_audit",
        "deviation_ledger",
        "paper_scale_numerical_result",
        "external_claim_slots",
        "artifact_manifest",
        "gates",
        "required_gates",
        "claim_boundary",
        "wall_time_seconds_finalize",
        "mutation_audit",
        "analysis_sha256",
    }
)

TRAINING_RUNTIME_SIGNATURE_KEYS = frozenset(
    {
        "schema_version",
        "python_executable",
        "python_version",
        "python_implementation",
        "sys_prefix",
        "conda_prefix",
        "conda_default_env",
        "platform",
        "machine",
        "numpy_version",
        "scipy_version",
        "pyarrow_version",
        "psutil_version",
        "torch_version",
        "torch_git_version",
        "torch_cuda_version",
        "cudnn_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_current_device",
        "cuda_device_names",
        "cuda_device_capabilities",
        "cuda_total_memory_bytes",
        "nvidia_smi_devices",
        "deterministic_algorithms_enabled",
        "deterministic_debug_mode",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
        "float32_matmul_precision",
        "torch_default_dtype",
        "torch_num_threads",
        "torch_num_interop_threads",
        "environment_controls",
    }
)
TRAINING_RUNTIME_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTHONHASHSEED",
        "KMP_DUPLICATE_LIB_OK",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    }
)

PROJECT_ACTION_ORDER = tuple(PARAMETER_NAMES)
NPZ_ATTEMPT_ID_FIELD = "__t9_1_3_attempt_id__"
GQF_ACTION_ORDER = (
    "virtual_rotation",
    "layer1_phi",
    "layer2_phi",
    "layer3_phi",
    "layer4_phi",
    "layer1_theta",
    "layer2_theta",
    "layer3_theta",
    "layer4_theta",
    "layer1_beta_real",
    "layer2_beta_real",
    "layer3_beta_real",
    "layer1_beta_imag",
    "layer2_beta_imag",
    "layer3_beta_imag",
)
_PROJECT_INDEX = {name: index for index, name in enumerate(PROJECT_ACTION_ORDER)}
_GQF_INDEX = {name: index for index, name in enumerate(GQF_ACTION_ORDER)}

STATE_SPEC = {
    "x_plus": ("X", 1.0),
    "x_minus": ("X", -1.0),
    "y_plus": ("Y", 1.0),
    "y_minus": ("Y", -1.0),
    "z_plus": ("Z", 1.0),
    "z_minus": ("Z", -1.0),
}
TRAJECTORY_ROW_KEYS = (
    "trajectory_id",
    "stratum",
    "strategy",
    "selected_agent_seed",
    "initial_state",
    "pauli_axis",
    "expected_eigenvalue",
    "evaluation_seed",
    "evaluation_cluster_id",
    "rng_stream_id",
    "batch_member",
    "state_definition_id",
    "logical_operator_definition_id",
    "raw_codeword_overlap_real",
    "raw_codeword_overlap_imag",
    "raw_logical_gram_condition_number",
    "cutoff",
    "full_cycles",
    "half_cycles",
    "timing_profile_id",
    "simulated_time_us",
    "projected_rho_00_real",
    "projected_rho_00_imag",
    "projected_rho_01_real",
    "projected_rho_01_imag",
    "projected_rho_10_real",
    "projected_rho_10_imag",
    "projected_rho_11_real",
    "projected_rho_11_imag",
    "joint_trace_error_max_for_batch",
    "joint_hermiticity_error_max_for_batch",
    "minimum_final_joint_eigenvalue_for_batch",
    "projected_hermiticity_error_before_stabilization",
    "code_weighted_I",
    "code_weighted_X",
    "code_weighted_Y",
    "code_weighted_Z",
    "signed_axis_signal",
    "state_fidelity_unconditional",
    "code_survival",
    "postselected",
    "included_in_denominator",
    "development_only",
    "censored",
    "status",
)
EVENT_ROW_KEYS = (
    "trajectory_id",
    "half_cycle",
    "full_cycle_coordinate",
    "layer_control_decision_prefix_length",
    "physical_layer_application_half_cycle",
    "measurement_outcome_half_cycle",
    "post_measurement_control_decision_prefix_length",
    "post_measurement_virtual_rotation_after_half_cycle",
    "post_measurement_control_layer_application_half_cycle",
    "uses_current_measurement_outcome_for_post_measurement_control",
    "terminal_feedback_correction",
    "outcome_g0_e1",
    "selected_branch_probability",
    "project_action_order_json",
    "gqf_action_order_json",
    "project_action_sha256",
    "post_measurement_project_action_order_json",
    "post_measurement_gqf_action_order_json",
    "post_measurement_project_action_sha256",
    "evaluation_cluster_id",
    "rng_stream_id",
    "postselected",
    "included_in_denominator",
    "development_only",
    "status",
)


class AgentArtifactError(ValueError):
    """Base class for resumable per-agent publication failures."""


class PartialAgentArtifactError(AgentArtifactError):
    """The commit-marker JSON is absent or the three-file bundle is partial."""


class CorruptAgentArtifactError(AgentArtifactError):
    """A same-contract bundle failed content/hash/reload validation."""


class AgentContractDriftError(AgentArtifactError):
    """A complete bundle belongs to different config/source bytes."""


class ConcurrentAgentWriterError(RuntimeError):
    """Another live process owns the same strategy/root publication lane."""


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T9.1.3 requires PyTorch; use "
            "C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _training_gpu_attestation_binding_valid(
    value: Any, config: Mapping[str, Any]
) -> bool:
    """Require a training-launch binding only for the frozen production config."""

    production = _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256
    if not production:
        return value is None
    try:
        binding = _validated_gpu_attestation_binding(value)
    except (GpuLoadAttestationError, TypeError, ValueError):
        return False
    return binding.get("purpose") == "TRAINING_LAUNCH"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_temporary_path(path: Path) -> Path:
    """Return a same-directory, process-unique path for atomic replacement."""

    return path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.{time.time_ns()}.tmp"
    )


def _replace_with_windows_retry(source: Path, destination: Path) -> None:
    """Retry the short sharing-violation window around concurrent replaces."""

    for attempt in range(100):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(min(0.001 * (attempt + 1), 0.02))


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    with _ATOMIC_JSON_PROCESS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _unique_temporary_path(path)
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _replace_with_windows_retry(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _atomic_npz(arrays: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        _replace_with_windows_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch(payload: Mapping[str, Any], path: Path) -> None:
    th = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(path)
    try:
        th.save(dict(payload), temporary)
        _replace_with_windows_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows must use one deterministic field order")
    temporary = _unique_temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        _replace_with_windows_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_roundtrip_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Normalize generated rows exactly as the on-disk CSV codec does."""

    if not rows:
        return []
    fieldnames = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return list(csv.DictReader(stream))


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Parquet shard: {path}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:  # pragma: no cover - DLEnv owns runtime.
        raise RuntimeError("T9.1.3 requires pyarrow for lossless raw shards") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(path)
    try:
        table = pa.Table.from_pylist([dict(row) for row in rows])
        pq.write_table(table, temporary, compression="zstd")
        _replace_with_windows_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def project_to_gqf_action(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Permute a trailing project-order action vector into public-GQF order."""

    array = np.asarray(values)
    if array.shape[-1] != len(PROJECT_ACTION_ORDER):
        raise ValueError("action vector must end in 15 parameters")
    return np.stack([array[..., _PROJECT_INDEX[name]] for name in GQF_ACTION_ORDER], axis=-1)


def gqf_to_project_action(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Permute a trailing public-GQF-order action vector into project order."""

    array = np.asarray(values)
    if array.shape[-1] != len(GQF_ACTION_ORDER):
        raise ValueError("action vector must end in 15 parameters")
    return np.stack([array[..., _GQF_INDEX[name]] for name in PROJECT_ACTION_ORDER], axis=-1)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _portable_artifact_path(path: Path) -> str:
    """Use repo-relative POSIX paths for portable production provenance."""

    root = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        # Unit/pilot runs may intentionally use an isolated pytest directory.
        return resolved.as_posix()


def _require_canonical_production_path(path: Path, relative: str) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() != (root / relative).resolve():
        raise ValueError(
            f"production artifact path must be {relative}, got {path}"
        )


def implementation_sha256() -> str:
    """Hash every executable source which defines this isolated lane."""

    root = Path(__file__).resolve().parents[2]
    paths = tuple(root / relative for relative in IMPLEMENTATION_RELATIVE_PATHS)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _reject_canonical_production_config_downgrade(
    config: Mapping[str, Any], *, production: bool
) -> None:
    """Reject the frozen workload before a nonproduction API can do any work."""

    if not production and _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256:
        raise ValueError(
            "the frozen 40x1000 production config cannot run with production=False"
        )


def _configure_production_determinism() -> None:
    """Enable the bit-exact production policy before any CUDA workload starts."""

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "production requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "production requires NVIDIA_TF32_OVERRIDE=0 before Python starts"
        )
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "0":
        raise RuntimeError(
            "production requires TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 before Python starts"
        )
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("production requires PYTHONHASHSEED=0 before Python starts")
    th = _require_torch()
    th.use_deterministic_algorithms(True, warn_only=False)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.allow_tf32 = False
    th.backends.cuda.matmul.allow_tf32 = False
    th.set_float32_matmul_precision("highest")


def _validate_production_determinism_signature(
    signature: Mapping[str, Any],
) -> None:
    """Fail closed unless the sealed runtime exposes the exact determinism policy."""

    controls = signature.get("environment_controls")
    if not isinstance(controls, Mapping):
        raise RuntimeError("production deterministic environment ledger is missing")
    expected_controls = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "PYTHONHASHSEED": "0",
    }
    mismatched = {
        name: controls.get(name)
        for name, expected in expected_controls.items()
        if controls.get(name) != expected
    }
    if mismatched:
        raise RuntimeError(
            "production deterministic environment controls drifted: "
            + ", ".join(f"{name}={value!r}" for name, value in mismatched.items())
        )
    if (
        signature.get("deterministic_algorithms_enabled") is not True
        or int(signature.get("deterministic_debug_mode", -1)) != 2
        or signature.get("cudnn_deterministic") is not True
        or signature.get("cudnn_benchmark") is not False
        or signature.get("cudnn_allow_tf32") is not False
        or signature.get("cuda_matmul_allow_tf32") is not False
        or signature.get("float32_matmul_precision") != "highest"
    ):
        raise RuntimeError("production torch deterministic/TF32 policy drifted")


def _validate_config(config: Mapping[str, Any], *, production: bool) -> None:
    if config.get("task_id") != TASK_ID:
        raise ValueError("config task_id mismatch")
    binding = config.get("protocol_binding", {})
    if binding.get("analysis_sha256") != T9_1_1_ANALYSIS_SHA256:
        raise ValueError("config does not bind the immutable T9.1.1 analysis hash")
    if config.get("evidence_grade") != EVIDENCE_GRADE:
        raise ValueError("evidence grade must remain paper-constrained")
    training = config.get("training")
    qualification = config.get("six_state_qualification")
    if not isinstance(training, dict) or not isinstance(qualification, dict):
        raise TypeError("training and six_state_qualification must be objects")
    roots = tuple(int(seed) for seed in training.get("paired_root_seeds", ()))
    validation = tuple(int(seed) for seed in training.get("validation_seeds", ()))
    evaluation = tuple(int(seed) for seed in qualification.get("seeds", ()))
    confirmation = tuple(int(seed) for seed in qualification.get("confirmation_seeds", ()))
    for name, values in (
        ("paired_root_seeds", roots),
        ("validation_seeds", validation),
        ("qualification.seeds", evaluation),
        ("qualification.confirmation_seeds", confirmation),
    ):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
    split_sets = [set(roots), set(validation), set(evaluation), set(confirmation)]
    for index, left in enumerate(split_sets):
        for right in split_sets[index + 1 :]:
            if left & right:
                raise ValueError("root/validation/qualification seed domains overlap")
    states = tuple(qualification.get("states", ()))
    if states != STATE_LABELS:
        raise ValueError(f"six states must be exactly {STATE_LABELS}")
    if int(training.get("validation_interval", 0)) <= 0:
        raise ValueError("validation_interval must be positive")
    if (
        training.get("feedback_timeline") != PAPER_CAUSAL_FEEDBACK_TIMELINE
        or training.get("observation_encoding") != "gqf_g_plus1_e_minus1"
        or training.get("consume_initial_zero_sentinel") is not True
    ):
        raise ValueError(
            "T9.1.3 must preserve the paper-causal GQF token/timeline contract"
        )
    if int(training.get("epochs", 0)) % int(training["validation_interval"]) != 0:
        raise ValueError("epochs must be divisible by validation_interval")
    epochs = int(training["epochs"])
    if training.get("epochs_by_strategy") != {"mf": epochs, "nmf": epochs}:
        raise ValueError("epochs_by_strategy must bind both families to training.epochs")
    selection = training.get("selection", {})
    if (
        selection.get("selection_data") != "validation_only"
        or selection.get("formal_or_qualification_data_visible_during_selection")
        is not False
        or selection.get("metric") != "mean_validation_final_state_fidelity"
        or selection.get("agent_level_best_of_twenty_on_test_or_lifetime")
        != "PROHIBITED"
    ):
        raise ValueError("checkpoint/agent selection must remain validation-only")
    required_checkpoints = set(
        training.get("checkpoint_contract", {}).get("required_per_agent", ())
    )
    checkpoint_minimum = {
        "validation_selected_model_state",
        "terminal_model_state",
        "optimizer_state_at_terminal_epoch",
        "torch_and_numpy_rng_state_at_terminal_epoch",
        "complete_epoch_metric_history",
        "complete_validation_candidate_ledger",
        "all_validation_candidate_model_states",
        "per_seed_and_per_trajectory_validation_final_fidelity",
        "training_runtime_signature_and_hash",
        "configuration_hash",
        "implementation_hash",
        "checkpoint_sha256",
    }
    if not checkpoint_minimum <= required_checkpoints:
        raise ValueError("checkpoint contract omits a required restart/provenance field")
    artifact_contract = config.get("artifact_contract", {})
    if (
        artifact_contract.get("environment_and_runtime_ledger_required") is not True
        or artifact_contract.get("runtime_manifest_scope")
        != (
            "direct imported dependency versions, Python executable/prefix, "
            "torch/CUDA/cuDNN flags, NVIDIA UUID/driver rows, thread settings "
            "and execution-semantic environment controls"
        )
        or artifact_contract.get("resume_identity")
        != [
            "protocol_analysis_sha256",
            "config_sha256",
            "implementation_sha256",
            "training_runtime_signature_sha256",
            "strategy",
            "paired_root_seed",
        ]
    ):
        raise ValueError("artifact runtime/resume identity contract drifted")
    if qualification.get("postselection") != "PROHIBITED":
        raise ValueError("six-state qualification must prohibit postselection")
    if (
        qualification.get("state_definition_id")
        != "LOWDIN_ORTHONORMALIZED_FINITE_CUTOFF_GRID_CODE_V1"
        or qualification.get("logical_operator_definition_id")
        != "PROJECTOR_DEFINED_CODE_PAULI_V1"
        or qualification.get("development_only") is not True
        or qualification.get("monte_carlo_design")
        != "paired_common_random_numbers_across_states_and_strategies_clustered_by_stratum_and_evaluation_seed"
    ):
        raise ValueError("six-state semantic and paired-cluster disclosure drifted")
    if tuple(qualification.get("strategies", ())) != (
        "standard",
        "selected_mf",
        "selected_nmf",
    ):
        raise ValueError("six-state qualification strategy set/order drifted")
    if (
        tuple(qualification.get("required_trajectory_fields", ()))
        != TRAJECTORY_ROW_KEYS
        or tuple(qualification.get("required_event_fields", ()))
        != EVENT_ROW_KEYS
    ):
        raise ValueError("six-state raw row exact schema drifted")
    if int(qualification.get("primary_cutoff", -1)) != int(training["cutoff"]):
        raise ValueError("qualification primary cutoff must equal training cutoff")
    if int(qualification.get("confirmation_cutoff", -1)) != int(
        training["confirmation_cutoff"]
    ):
        raise ValueError("qualification confirmation cutoff mismatch")
    paper_target = config.get("paper_scale_target", {})
    if paper_target.get("status") != "NOT_EVALUATED_NULL" or any(
        value is not None for value in paper_target.get("numeric_values", {}).values()
    ):
        raise ValueError("paper-scale target values must remain explicitly null")
    paper_result = config.get("paper_scale_result", {})
    if (
        paper_result.get("state") != PAPER_NUMERIC_STATE
        or paper_result.get("value") is not None
    ):
        raise ValueError("paper-scale result must remain incomplete/null")
    external_slots = config.get("external_claim_slots", {})
    if any(
        external_slots.get(name, {}).get("value") is not None
        for name in ("official_exact", "surpass")
    ):
        raise ValueError("external exact/surpass claim slots must remain null")
    _validated_deviation_ledger(config)
    if production:
        if _canonical_sha256(config) != PRODUCTION_CONFIG_SHA256:
            raise ValueError("production config bytes/values differ from the frozen seal")
        expected = {
            "epochs": 1000,
            "full_cycles": 10,
            "learning_rate": 1.0e-4,
            "score_baseline_decay": 0.95,
            "gradient_clip_norm": 10.0,
            "residual_l2_weight": 1.0e-5,
            "slew_l2_weight": 1.0e-5,
            "projector_delta": 0.2,
            "noise": "high",
            "cutoff": 12,
            "confirmation_cutoff": 16,
            "mf_batch_size": 8,
            "nmf_batch_size": 6,
            "validation_batch_size": 32,
            "device": "cuda",
            "real_dtype": "float64",
        }
        for field, wanted in expected.items():
            if training.get(field) != wanted:
                raise ValueError(f"production {field} must equal {wanted!r}")
        if len(roots) != 20:
            raise ValueError("production requires exactly 20 paired root seeds")
        if int(qualification.get("full_cycles", 0)) != 10:
            raise ValueError("qualification must remain ten full cycles")
        if paper_target.get("fock_cutoff") != 100:
            raise ValueError("paper-scale target must retain cutoff 100")
        if paper_target.get("evaluation_full_cycles") != 1000:
            raise ValueError("paper-scale target must retain 1000 cycles")


def _verify_parent_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["protocol_binding"]
    path = _resolve_repo_path(binding["path"])
    payload = _load_json(path)
    from cnn_fpga.benchmark.phase9_three_lane_protocol import verify_report

    checks = verify_report(report=payload, path=path)
    actual = payload.get("analysis_sha256")
    if actual != binding["analysis_sha256"] or actual != T9_1_1_ANALYSIS_SHA256:
        raise ValueError("live T9.1.1 protocol analysis hash mismatch")
    return {
        "path": binding["path"],
        "analysis_sha256": actual,
        "file_sha256": _file_sha256(path),
        "verification_checks": checks,
    }


def _validated_deviation_ledger(
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Validate the full D01--D42 register, not only a row count."""

    path = _resolve_repo_path(config["deviation_ledger_path"])
    if (
        _portable_artifact_path(path)
        != "configs/phase9/t9_1_3_deviation_ledger.json"
        or _file_sha256(path) != DEVIATION_LEDGER_SHA256
    ):
        raise ValueError("T9.1.3 deviation ledger path/hash drifted")
    payload = _load_json(path)
    if set(payload) != {
        "schema_version",
        "task_id",
        "frozen_date",
        "protocol_binding",
        "configuration_binding",
        "source_binding",
        "field_contract",
        "deviations",
        "summary",
    }:
        raise ValueError("deviation ledger top-level schema drifted")
    if (
        payload.get("schema_version")
        != "t9.1.3-puviani-deviation-ledger-v1"
        or payload.get("task_id") != TASK_ID
        or payload.get("frozen_date") != "2026-07-22"
        or payload.get("configuration_binding")
        != "configs/phase9/t9_1_3_puviani_paper_constrained.json"
        or payload.get("protocol_binding")
        != {
            "path": "docs/t9_1_1_three_lane_protocol.json",
            "analysis_sha256": T9_1_1_ANALYSIS_SHA256,
        }
    ):
        raise ValueError("deviation ledger identity/binding drifted")
    root = Path(__file__).resolve().parents[2]
    paper_path = (
        "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
        "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
    )
    source_binding = payload.get("source_binding")
    if (
        source_binding
        != {
            "paper_path": paper_path,
            "paper_sha256": "5dcda2ee0fec3191ddaa44fb539764dbebc2d089c9a3818bb17c132ca28e7c22",
            "official_repository": "https://github.com/Matteo-Puviani/GQF.git",
            "gqf_commit": GQF_COMMIT,
            "inherited_discrepancy_source": "docs/t6_8_4_gqf_paper_exact_reproduction.json",
        }
        or _file_sha256(root / paper_path) != source_binding["paper_sha256"]
        or not (root / source_binding["inherited_discrepancy_source"]).is_file()
    ):
        raise ValueError("deviation ledger source binding drifted")
    required_fields = [
        "id",
        "topic",
        "locator",
        "value",
        "chosen",
        "rationale",
        "severity",
        "comparability",
        "paper_numeric_anchor_eligible",
        "claim_effect",
    ]
    if payload.get("field_contract") != {
        "required_per_entry": required_fields,
        "severity_order": ["Low", "Medium", "High", "Critical"],
        "missing_value_semantics": (
            "null means unavailable or unspecified; it never means zero or agreement"
        ),
        "allowed_evidence_label": EVIDENCE_GRADE,
    }:
        raise ValueError("deviation ledger field contract drifted")
    deviations = payload.get("deviations")
    if not isinstance(deviations, list) or len(deviations) != 42:
        raise ValueError("deviation ledger must contain exactly D01--D42")
    expected_ids = [f"D{index:02d}" for index in range(1, 43)]
    if [row.get("id") for row in deviations] != expected_ids:
        raise ValueError("deviation ledger ID/order census drifted")
    for index, row in enumerate(deviations, start=1):
        expected_keys = set(required_fields) | ({"inherited_from"} if index <= 18 else set())
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError(f"deviation D{index:02d} schema drifted")
        if index <= 18 and row.get("inherited_from") != f"T6.8.4:D{index:02d}":
            raise ValueError(f"deviation D{index:02d} inheritance drifted")
        if (
            not all(
                isinstance(row.get(name), str) and bool(row[name].strip())
                for name in (
                    "topic",
                    "chosen",
                    "rationale",
                    "comparability",
                    "claim_effect",
                )
            )
            or row.get("severity") not in {"Low", "Medium", "High", "Critical"}
            or row.get("paper_numeric_anchor_eligible") is not False
            or not isinstance(row.get("value"), Mapping)
            or not isinstance(row.get("locator"), Mapping)
            or set(row["locator"]) != {"paper", "official", "project"}
            or any(
                value is not None and (not isinstance(value, str) or not value.strip())
                for value in row["locator"].values()
            )
        ):
            raise ValueError(f"deviation D{index:02d} content contract drifted")
    counts = Counter(str(row["severity"]) for row in deviations)
    expected_summary = {
        "entry_count": 42,
        "inherited_D01_through_D18_present": True,
        "critical_count": 23,
        "high_count": 16,
        "medium_count": 3,
        "low_count": 0,
        "official_exact_reproduction": "BLOCKED",
        "paper_constrained_artifact_generation": "ALLOWED_WITH_ALL_DEVIATIONS_ATTACHED",
        "paper_scale_numeric_results": "NOT_EVALUATED_NULL",
        "permitted_claim": EVIDENCE_GRADE,
    }
    if (
        payload.get("summary") != expected_summary
        or counts != Counter({"Critical": 23, "High": 16, "Medium": 3})
    ):
        raise ValueError("deviation ledger summary/severity census drifted")
    return path, payload, deviations


def _strategy_config(config: Mapping[str, Any], strategy: Literal["mf", "nmf"]) -> DirectionalRankingConfig:
    training = config["training"]
    qualification = config["six_state_qualification"]
    return DirectionalRankingConfig(
        cutoff=int(training["cutoff"]),
        confirmation_cutoff=int(training["confirmation_cutoff"]),
        full_cycles=int(training["full_cycles"]),
        train_epochs=int(training["epochs"]),
        train_batch_size=int(training[f"{strategy}_batch_size"]),
        validation_batch_size=int(training["validation_batch_size"]),
        test_batch_size=int(qualification["batch_size"]),
        confirmation_batch_size=int(qualification["confirmation_batch_size"]),
        validation_interval=int(training["validation_interval"]),
        learning_rate=float(training["learning_rate"]),
        score_baseline_decay=float(training.get("score_baseline_decay", 0.95)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 10.0)),
        residual_l2_weight=float(training.get("residual_l2_weight", 1.0e-5)),
        slew_l2_weight=float(training.get("slew_l2_weight", 1.0e-5)),
        training_seeds=tuple(int(seed) for seed in training["paired_root_seeds"]),
        validation_seeds=tuple(int(seed) for seed in training["validation_seeds"]),
        test_seeds=tuple(int(seed) for seed in qualification["seeds"]),
        confirmation_seeds=tuple(int(seed) for seed in qualification["confirmation_seeds"]),
        bootstrap_seed=int(training["bootstrap_seed"]),
        bootstrap_repetitions=int(training.get("bootstrap_repetitions", 50_000)),
        device=str(training["device"]),
        real_dtype=str(training["real_dtype"]),
        projector_delta=float(training["projector_delta"]),
        cavity_lifetime_us=float(training["noise_parameters_us"]["cavity_lifetime"]),
        ancilla_t1_us=float(training["noise_parameters_us"]["ancilla_t1"]),
        ancilla_t2_us=float(training["noise_parameters_us"]["ancilla_t2"]),
        feedback_timeline=str(training["feedback_timeline"]),
        observation_encoding=str(training["observation_encoding"]),
        consume_initial_zero_sentinel=bool(
            training["consume_initial_zero_sentinel"]
        ),
    )


def _state_numpy(model: Any) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().contiguous().numpy()
        for name, tensor in model.state_dict().items()
    }


def _numpy_state_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_npz_model(
    path: Path,
    strategy: Literal["mf", "nmf"],
    config: DirectionalRankingConfig,
    seed: int,
    *,
    expected_attempt_id: str | None = None,
) -> tuple[Any, str]:
    th = _require_torch()
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, EOFError) as error:
        raise CorruptAgentArtifactError(f"checkpoint archive cannot be read: {path}") from error
    attempt_value = arrays.pop(NPZ_ATTEMPT_ID_FIELD, None)
    if (
        attempt_value is None
        or np.asarray(attempt_value).shape != ()
        or not str(np.asarray(attempt_value).item())
        or (
            expected_attempt_id is not None
            and str(np.asarray(attempt_value).item()) != expected_attempt_id
        )
    ):
        raise CorruptAgentArtifactError(
            f"checkpoint attempt identity mismatch: {path}"
        )
    model = build_policy(strategy, config, seed)
    reference = model.state_dict()
    if set(arrays) != set(reference):
        raise CorruptAgentArtifactError(f"checkpoint tensor names mismatch: {path}")
    for name, tensor in reference.items():
        value = arrays[name]
        expected_dtype = tensor.detach().cpu().numpy().dtype
        if (
            value.shape != tuple(tensor.shape)
            or value.dtype != expected_dtype
            or not np.all(np.isfinite(value))
        ):
            raise CorruptAgentArtifactError(
                f"checkpoint tensor dtype/shape/finiteness mismatch: {path}:{name}"
            )
    state = {
        name: th.as_tensor(arrays[name], dtype=tensor.dtype, device=tensor.device)
        for name, tensor in reference.items()
    }
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise CorruptAgentArtifactError(f"checkpoint tensor shapes mismatch: {path}") from error
    model.eval()
    return model, _numpy_state_sha256(arrays)


def _agent_paths(output_dir: Path, strategy: str, seed: int) -> tuple[Path, Path, Path]:
    directory = output_dir / strategy
    stem = f"{strategy}_seed_{seed}"
    return (
        directory / f"{stem}.npz",
        directory / f"{stem}.terminal.pt",
        directory / f"{stem}.json",
    )


def _new_attempt_directory(
    output_dir: Path, category: str, strategy: str, seed: int
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (
        output_dir
        / category
        / f"{strategy}_seed_{seed}_{stamp}_{os.getpid()}_{time.time_ns()}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _attempt_id(directory: Path) -> str:
    """Use the collision-resistant directory basename as the immutable attempt ID."""

    value = directory.name
    if not value or directory.parent.name != "_training_attempts":
        raise ValueError("attempt ID must originate from a training-attempt directory")
    return value


def _archive_incomplete_agent_bundle(
    *,
    output_dir: Path,
    strategy: str,
    seed: int,
    config: Mapping[str, Any],
    implementation_hash: str,
    error: BaseException,
    traceback_text: str,
) -> Path:
    """Preserve every partial byte before a clean same-agent retry."""

    active_successes = _active_success_attempt_directories(
        output_dir, strategy, seed
    )
    if len(active_successes) > 1:
        raise AgentArtifactError(
            f"multiple active success attempts exist for {strategy} seed {seed}"
        )
    archive = _new_attempt_directory(
        output_dir, "_corrupt_attempts", strategy, seed
    )
    failure_path = archive / "failure.json"
    runtime_signature, runtime_signature_hash = _current_training_runtime()
    base_failure = {
        "schema_version": "t9.1.3-agent-failure-v1",
        "task_id": TASK_ID,
        "strategy": strategy,
        "root_seed": int(seed),
        "failure_state": "NULL",
        "failure_subtype": "CORRUPT_SHARD_ARCHIVE_IN_PROGRESS",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback_text,
        "config_sha256": _canonical_sha256(config),
        "agent_contract_hash": _agent_contract_hash(
            config, strategy, seed, implementation_hash, runtime_signature
        ),
        "training_runtime_signature": runtime_signature,
        "training_runtime_signature_sha256": runtime_signature_hash,
        "implementation_sha256": implementation_hash,
        "parent_analysis_sha256": T9_1_1_ANALYSIS_SHA256,
        "archived_partial_files": [],
        "retained_instead_of_silent_drop": True,
        "safe_retry_authorized": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Commit a provisional marker before the first move.  Startup recovery can
    # therefore finish even if this quarantine operation itself is interrupted.
    _atomic_json(base_failure, failure_path)
    archived: list[dict[str, Any]] = []
    for canonical_path in _agent_paths(output_dir, strategy, seed):
        # A hard process termination can occur before or after os.replace.  Keep
        # both the last published name and its temporary predecessor so a retry
        # never silently overwrites forensic bytes.
        temporary_candidates = sorted(
            canonical_path.parent.glob(f".{canonical_path.name}.*.tmp")
        )
        for source in (canonical_path, *temporary_candidates):
            if source.exists():
                destination = archive / source.name
                os.replace(source, destination)
                archived.append(
                    {
                        "path": _portable_artifact_path(destination),
                        "sha256": _file_sha256(destination),
                        "bytes": destination.stat().st_size,
                    }
                )
    base_failure.update(
        {
            "failure_state": "CORRUPT_SHARD" if archived else "NULL",
            "failure_subtype": (
                "PARTIAL_OR_CORRUPT_BUNDLE"
                if archived
                else "PUBLICATION_FAILURE_BEFORE_ANY_SHARD"
            ),
            "archived_partial_files": archived,
            "safe_retry_authorized": True,
            "archive_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(base_failure, failure_path)
    if active_successes:
        _mark_attempt_superseded(
            active_successes[0], failure_path, strategy=strategy, seed=seed
        )
    return failure_path


def _uncached_training_runtime_signature() -> dict[str, Any]:
    """Capture the immutable software/device semantics used to train agents."""

    th = _require_torch()
    try:
        import scipy
    except ModuleNotFoundError:  # pragma: no cover - production validation rejects it.
        scipy_version = None
    else:
        scipy_version = scipy.__version__
    try:
        import pyarrow
    except ModuleNotFoundError:  # pragma: no cover - production validation rejects it.
        pyarrow_version = None
    else:
        pyarrow_version = pyarrow.__version__
    cuda_available = bool(th.cuda.is_available())
    device_count = int(th.cuda.device_count()) if cuda_available else 0
    driver_devices: list[dict[str, str]] = []
    if cuda_available:
        import subprocess

        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            for line in completed.stdout.splitlines():
                values = [value.strip() for value in line.split(",", 1)]
                if len(values) == 2:
                    driver_devices.append(
                        {"uuid": values[0], "driver_version": values[1]}
                    )
        except (OSError, subprocess.SubprocessError):
            driver_devices = []
    return {
        "schema_version": "t9.1.3-training-runtime-signature-v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "pyarrow_version": pyarrow_version,
        "psutil_version": getattr(psutil, "__version__", None),
        "torch_version": th.__version__,
        "torch_git_version": getattr(th.version, "git_version", None),
        "torch_cuda_version": th.version.cuda,
        "cudnn_version": th.backends.cudnn.version() if cuda_available else None,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_current_device": int(th.cuda.current_device()) if cuda_available else None,
        "cuda_device_names": [th.cuda.get_device_name(index) for index in range(device_count)],
        "cuda_device_capabilities": [list(th.cuda.get_device_capability(index)) for index in range(device_count)],
        "cuda_total_memory_bytes": [int(th.cuda.get_device_properties(index).total_memory) for index in range(device_count)],
        "nvidia_smi_devices": driver_devices,
        "deterministic_algorithms_enabled": bool(th.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": int(th.get_deterministic_debug_mode()),
        "cudnn_enabled": bool(th.backends.cudnn.enabled),
        "cudnn_deterministic": bool(th.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(th.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(th.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(th.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": th.get_float32_matmul_precision(),
        "torch_default_dtype": str(th.get_default_dtype()),
        "torch_num_threads": int(th.get_num_threads()),
        "torch_num_interop_threads": int(th.get_num_interop_threads()),
        "environment_controls": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTHONHASHSEED",
                "KMP_DUPLICATE_LIB_OK",
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            )
        },
    }


def _training_runtime_signature() -> dict[str, Any]:
    # Re-read execution-semantic flags and environment controls on every
    # contract check; caching would hide an in-process runtime mutation.
    return _uncached_training_runtime_signature()


def _validated_training_runtime_signature(
    value: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], str]:
    """Validate and canonicalize a stored training/finalization runtime ledger."""

    if not isinstance(value, Mapping):
        raise CorruptAgentArtifactError(f"{label} must be a mapping")
    signature = dict(value)
    if set(signature) != TRAINING_RUNTIME_SIGNATURE_KEYS:
        raise CorruptAgentArtifactError(f"{label} schema drifted")
    if signature.get("schema_version") != "t9.1.3-training-runtime-signature-v1":
        raise CorruptAgentArtifactError(f"{label} version drifted")
    required_strings = (
        "python_executable",
        "python_version",
        "python_implementation",
        "sys_prefix",
        "platform",
        "machine",
        "numpy_version",
        "torch_version",
        "float32_matmul_precision",
        "torch_default_dtype",
    )
    if any(
        not isinstance(signature.get(name), str) or not signature[name]
        for name in required_strings
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid required string")
    optional_strings = (
        "conda_prefix",
        "conda_default_env",
        "scipy_version",
        "pyarrow_version",
        "psutil_version",
        "torch_git_version",
        "torch_cuda_version",
    )
    if any(
        signature.get(name) is not None
        and not isinstance(signature.get(name), str)
        for name in optional_strings
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid optional string")
    cudnn_version = signature.get("cudnn_version")
    if cudnn_version is not None and (
        isinstance(cudnn_version, bool) or not isinstance(cudnn_version, int)
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid cuDNN version")
    bool_fields = (
        "cuda_available",
        "deterministic_algorithms_enabled",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
    )
    if any(not isinstance(signature.get(name), bool) for name in bool_fields):
        raise CorruptAgentArtifactError(f"{label} has an invalid boolean field")
    integer_fields = (
        "cuda_device_count",
        "deterministic_debug_mode",
        "torch_num_threads",
        "torch_num_interop_threads",
    )
    if any(
        isinstance(signature.get(name), bool)
        or not isinstance(signature.get(name), int)
        for name in integer_fields
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid integer field")
    if (
        int(signature["cuda_device_count"]) < 0
        or int(signature["torch_num_threads"]) <= 0
        or int(signature["torch_num_interop_threads"]) <= 0
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid runtime count")
    device_count = int(signature["cuda_device_count"])
    current_device = signature.get("cuda_current_device")
    if current_device is not None and (
        isinstance(current_device, bool) or not isinstance(current_device, int)
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid CUDA device index")
    vector_fields = (
        "cuda_device_names",
        "cuda_device_capabilities",
        "cuda_total_memory_bytes",
    )
    if any(not isinstance(signature.get(name), list) for name in vector_fields):
        raise CorruptAgentArtifactError(f"{label} has an invalid CUDA device vector")
    if any(len(signature[name]) != device_count for name in vector_fields):
        raise CorruptAgentArtifactError(f"{label} CUDA vector census drifted")
    if any(not isinstance(name, str) or not name for name in signature["cuda_device_names"]):
        raise CorruptAgentArtifactError(f"{label} has an invalid CUDA device name")
    if any(
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
        for capability in signature["cuda_device_capabilities"]
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid CUDA capability")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in signature["cuda_total_memory_bytes"]
    ):
        raise CorruptAgentArtifactError(f"{label} has an invalid CUDA memory census")
    if bool(signature["cuda_available"]):
        if device_count <= 0 or current_device not in range(device_count):
            raise CorruptAgentArtifactError(f"{label} CUDA availability is inconsistent")
    elif device_count != 0 or current_device is not None:
        raise CorruptAgentArtifactError(f"{label} CPU-only CUDA census is inconsistent")
    driver_devices = signature.get("nvidia_smi_devices")
    if not isinstance(driver_devices, list):
        raise CorruptAgentArtifactError(f"{label} NVIDIA driver ledger is invalid")
    for row in driver_devices:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"uuid", "driver_version"}
            or not isinstance(row.get("uuid"), str)
            or not row["uuid"]
            or not isinstance(row.get("driver_version"), str)
            or not row["driver_version"]
        ):
            raise CorruptAgentArtifactError(f"{label} NVIDIA driver row is invalid")
    controls = signature.get("environment_controls")
    if not isinstance(controls, Mapping) or set(controls) != TRAINING_RUNTIME_ENVIRONMENT_KEYS:
        raise CorruptAgentArtifactError(f"{label} environment-control schema drifted")
    if any(value is not None and not isinstance(value, str) for value in controls.values()):
        raise CorruptAgentArtifactError(f"{label} environment-control value is invalid")
    return signature, _canonical_sha256(signature)


def _current_training_runtime() -> tuple[dict[str, Any], str]:
    signature = _training_runtime_signature()
    return _validated_training_runtime_signature(
        signature, label="current training runtime signature"
    )


def _validate_runtime_for_config(
    config: Mapping[str, Any], *, production: bool
) -> tuple[dict[str, Any], str]:
    signature, digest = _current_training_runtime()
    artifact_contract = config.get("artifact_contract", {})
    if artifact_contract.get("environment_and_runtime_ledger_required") is not True:
        raise ValueError("artifact contract must require the runtime ledger")
    if production:
        _validate_production_determinism_signature(signature)
        missing = [
            name
            for name in ("scipy_version", "pyarrow_version", "psutil_version")
            if signature.get(name) is None
        ]
        if missing:
            raise RuntimeError(
                "production runtime omits required packages: " + ", ".join(missing)
            )
        if (
            config["training"].get("device") != "cuda"
            or signature.get("cuda_available") is not True
            or int(signature.get("cuda_device_count", 0)) <= 0
            or signature.get("torch_cuda_version") is None
        ):
            raise RuntimeError("production training requires a live CUDA torch runtime")
        driver_rows = signature.get("nvidia_smi_devices", [])
        driver_uuids = [row["uuid"] for row in driver_rows]
        if (
            len(driver_rows) < int(signature["cuda_device_count"])
            or len(driver_uuids) != len(set(driver_uuids))
        ):
            raise RuntimeError(
                "production runtime requires a unique NVIDIA UUID/driver ledger"
            )
    return signature, digest


def _agent_contract_hash(
    config: Mapping[str, Any],
    strategy: str,
    seed: int,
    implementation_hash: str,
    runtime_signature: Mapping[str, Any] | None = None,
) -> str:
    signature = (
        dict(runtime_signature)
        if runtime_signature is not None
        else _training_runtime_signature()
    )
    signature, signature_hash = _validated_training_runtime_signature(
        signature, label=f"agent contract {strategy} seed {seed}"
    )
    if _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256:
        _validate_production_determinism_signature(signature)
    return _canonical_sha256(
        {
            "schema": SCHEMA_VERSION,
            "config": config,
            "strategy": strategy,
            "seed": int(seed),
            "implementation_sha256": implementation_hash,
            "parent_analysis_sha256": T9_1_1_ANALYSIS_SHA256,
            "training_runtime_signature_sha256": signature_hash,
        }
    )


def _failure_record(
    *,
    config: Mapping[str, Any],
    implementation_hash: str,
    strategy: str,
    seed: int,
    failure_state: str,
    failure_subtype: str,
    exception_type: str,
    exception_message: str,
    traceback_text: str,
    archived_partial_files: Sequence[Mapping[str, Any]] = (),
    safe_retry_authorized: bool,
    runtime_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_signature = (
        dict(runtime_signature)
        if runtime_signature is not None
        else _training_runtime_signature()
    )
    runtime_signature, runtime_signature_hash = _validated_training_runtime_signature(
        runtime_signature, label=f"failure record {strategy} seed {seed}"
    )
    if _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256:
        _validate_production_determinism_signature(runtime_signature)
    return {
        "schema_version": "t9.1.3-agent-failure-v1",
        "task_id": TASK_ID,
        "strategy": strategy,
        "root_seed": int(seed),
        "failure_state": failure_state,
        "failure_subtype": failure_subtype,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "traceback": traceback_text,
        "config_sha256": _canonical_sha256(config),
        "agent_contract_hash": _agent_contract_hash(
            config, strategy, seed, implementation_hash, runtime_signature
        ),
        "training_runtime_signature": runtime_signature,
        "training_runtime_signature_sha256": runtime_signature_hash,
        "implementation_sha256": implementation_hash,
        "parent_analysis_sha256": T9_1_1_ANALYSIS_SHA256,
        "archived_partial_files": [dict(row) for row in archived_partial_files],
        "retained_instead_of_silent_drop": True,
        "safe_retry_authorized": bool(safe_retry_authorized),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _archive_file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": _portable_artifact_path(path),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _parse_attempt_identity(directory: Path) -> tuple[str, int]:
    try:
        strategy, tail = directory.name.split("_seed_", 1)
        seed = int(tail.split("_", 1)[0])
    except (ValueError, IndexError) as error:
        raise AgentArtifactError(
            f"unparseable attempt directory identity: {directory}"
        ) from error
    if strategy not in {"mf", "nmf"}:
        raise AgentArtifactError(f"unknown attempt strategy: {directory}")
    return strategy, seed


def _active_success_attempt_directories(
    output_dir: Path, strategy: str, seed: int
) -> list[Path]:
    root = output_dir / "_training_attempts"
    if not root.exists():
        return []
    active: list[Path] = []
    for directory in sorted(
        path
        for path in root.glob(f"{strategy}_seed_{seed}_*")
        if path.is_dir()
    ):
        outcome_path = directory / "outcome.json"
        if not outcome_path.is_file() or (directory / "superseded.json").exists():
            continue
        outcome = _load_json(outcome_path)
        start_path = directory / "start.json"
        if not start_path.is_file():
            continue
        start = _load_json(start_path)
        if (
            outcome.get("state") == "SUCCESS"
            and outcome.get("attempt_id") == _attempt_id(directory)
            and start.get("attempt_id") == _attempt_id(directory)
            and start.get("runtime_signature_state")
            == "CAPTURED_BEFORE_TRAINING"
            and outcome.get("training_runtime_signature_sha256")
            == start.get("training_runtime_signature_sha256")
            and outcome.get("agent_contract_hash")
            == start.get("agent_contract_hash")
        ):
            active.append(directory)
    return active


def _mark_attempt_superseded(
    attempt_directory: Path,
    archive_failure_path: Path,
    *,
    strategy: str,
    seed: int,
) -> None:
    start = _load_json(attempt_directory / "start.json")
    _atomic_json(
        {
            "schema_version": "t9.1.3-attempt-superseded-v1",
            "task_id": TASK_ID,
            "strategy": strategy,
            "root_seed": int(seed),
            "attempt_id": _attempt_id(attempt_directory),
            "state": "SUPERSEDED_CORRUPT",
            "archive_failure_path": _portable_artifact_path(archive_failure_path),
            "archive_failure_sha256": _file_sha256(archive_failure_path),
            "training_runtime_signature_sha256": start.get(
                "training_runtime_signature_sha256"
            ),
            "superseded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        attempt_directory / "superseded.json",
    )


def _recover_incomplete_archives(
    output_dir: Path,
    config: Mapping[str, Any],
    implementation_hash: str,
    *,
    strategy_filter: str | None = None,
    seed_filter: int | None = None,
) -> None:
    root = output_dir / "_corrupt_attempts"
    if not root.exists():
        return
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        strategy, seed = _parse_attempt_identity(directory)
        if strategy_filter is not None and strategy != strategy_filter:
            continue
        if seed_filter is not None and seed != int(seed_filter):
            continue
        failure_path = directory / "failure.json"
        payload: dict[str, Any] | None = None
        if failure_path.exists():
            try:
                payload = _load_json(failure_path)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                retained = directory / f"failure_payload_corrupt_{time.time_ns()}.json"
                os.replace(failure_path, retained)
        if payload is not None and (
            payload.get("safe_retry_authorized") is True
            and payload.get("failure_subtype")
            != "CORRUPT_SHARD_ARCHIVE_IN_PROGRESS"
        ):
            continue
        archived = [
            _archive_file_entry(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
            and path != failure_path
            and path.name != "failure.json.tmp"
        ]
        recovered = _failure_record(
            config=config,
            implementation_hash=implementation_hash,
            strategy=strategy,
            seed=seed,
            failure_state="CORRUPT_SHARD" if archived else "NULL",
            failure_subtype="ARCHIVE_INTERRUPTED_AND_RECOVERED",
            exception_type=(payload or {}).get(
                "exception_type", "InterruptedArchiveRecovery"
            ),
            exception_message=(payload or {}).get(
                "exception_message", "archive marker was absent or incomplete"
            ),
            traceback_text=(payload or {}).get("traceback", ""),
            archived_partial_files=archived,
            safe_retry_authorized=True,
        )
        recovered["archive_recovered_after_interruption"] = True
        _atomic_json(recovered, failure_path)
        active = _active_success_attempt_directories(output_dir, strategy, seed)
        canonical_names = {
            path.name for path in _agent_paths(output_dir, strategy, seed)
        }
        if archived and any(
            Path(str(entry["path"])).name in canonical_names for entry in archived
        ):
            if len(active) > 1:
                raise AgentArtifactError(
                    f"archive recovery found multiple active successes for {strategy} seed {seed}"
                )
            # A first publication can be interrupted after one canonical file
            # was moved but before any successful attempt existed.  That is a
            # valid retained failure followed by a retry, not an ambiguity.
            if active:
                _mark_attempt_superseded(
                    active[0], failure_path, strategy=strategy, seed=seed
                )


def _is_atomic_temporary(path: Path) -> bool:
    return path.is_file() and path.name.startswith(".") and path.name.endswith(".tmp")


def _recover_incomplete_failed_attempts(
    output_dir: Path,
    config: Mapping[str, Any],
    implementation_hash: str,
    *,
    strategy_filter: str | None = None,
    seed_filter: int | None = None,
) -> None:
    """Make every interrupted retained-failure directory self-describing."""

    root = output_dir / "_failed_attempts"
    if not root.exists():
        return
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        strategy, seed = _parse_attempt_identity(directory)
        if strategy_filter is not None and strategy != strategy_filter:
            continue
        if seed_filter is not None and seed != int(seed_filter):
            continue
        failure_path = directory / "failure.json"
        payload: dict[str, Any] | None = None
        if failure_path.is_file():
            try:
                payload = _load_json(failure_path)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                retained = directory / f"failure_payload_corrupt_{time.time_ns()}.json"
                os.replace(failure_path, retained)
        all_other_files = {
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path != failure_path
        }
        declared: set[Path] = set()
        if payload is not None:
            try:
                declared = {
                    _resolve_repo_path(str(entry["path"])).resolve()
                    for entry in payload.get("archived_partial_files", [])
                }
            except (KeyError, TypeError, ValueError):
                declared = set()
        if (
            payload is not None
            and payload.get("safe_retry_authorized") is True
            and declared == all_other_files
        ):
            continue
        if failure_path.is_file():
            retained = directory / f"failure_payload_retained_{time.time_ns()}.json"
            os.replace(failure_path, retained)
            all_other_files.add(retained.resolve())
        archived = [_archive_file_entry(path) for path in sorted(all_other_files)]
        recovered = _failure_record(
            config=config,
            implementation_hash=implementation_hash,
            strategy=strategy,
            seed=seed,
            failure_state="CORRUPT_SHARD" if archived else "NULL",
            failure_subtype="FAILED_ATTEMPT_ARCHIVE_INTERRUPTED_AND_RECOVERED",
            exception_type="InterruptedFailureArchive",
            exception_message=(
                "retained failure directory existed without a complete authoritative marker"
            ),
            traceback_text="",
            archived_partial_files=archived,
            safe_retry_authorized=True,
        )
        recovered["archive_recovered_after_interruption"] = True
        _atomic_json(recovered, failure_path)


def _quarantine_agent_atomic_temporaries(
    *,
    output_dir: Path,
    strategy: str,
    seed: int,
    config: Mapping[str, Any],
    implementation_hash: str,
) -> Path | None:
    """Retain hard-kill atomic-write remnants before resuming one agent."""

    candidates: set[Path] = set()
    for canonical in _agent_paths(output_dir, strategy, seed):
        candidates.update(canonical.parent.glob(f".{canonical.name}.*.tmp"))
    pattern = f"{strategy}_seed_{seed}_*"
    for category in ("_training_attempts", "_corrupt_attempts", "_failed_attempts"):
        category_root = output_dir / category
        if not category_root.exists():
            continue
        for directory in category_root.glob(pattern):
            if directory.is_dir():
                candidates.update(path for path in directory.rglob("*") if _is_atomic_temporary(path))
    candidates = {path for path in candidates if path.is_file()}
    if not candidates:
        return None
    archive = _new_attempt_directory(output_dir, "_failed_attempts", strategy, seed)
    moved: list[dict[str, Any]] = []
    sources: list[str] = []
    for index, source in enumerate(sorted(candidates)):
        try:
            relative = source.resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            relative = source.as_posix()
        destination = archive / f"atomic_temp_{index:04d}_{source.name}"
        os.replace(source, destination)
        moved.append(_archive_file_entry(destination))
        sources.append(relative)
    failure = _failure_record(
        config=config,
        implementation_hash=implementation_hash,
        strategy=strategy,
        seed=seed,
        failure_state="CORRUPT_SHARD",
        failure_subtype="ORPHAN_ATOMIC_TEMP_RECOVERED",
        exception_type="InterruptedAtomicReplace",
        exception_message="hard-kill atomic-write remnants were retained before retry",
        traceback_text="",
        archived_partial_files=moved,
        safe_retry_authorized=True,
    )
    failure["orphan_atomic_sources"] = sources
    _atomic_json(failure, archive / "failure.json")
    return archive / "failure.json"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _process_created_unix_ns(pid: int) -> int | None:
    """Return an OS-backed process creation identity when it is available."""

    if pid <= 0 or psutil is None:
        return None
    try:
        return int(round(float(psutil.Process(pid).create_time()) * 1_000_000_000))
    except (psutil.Error, OSError, ValueError):
        return None


def _local_owner_is_alive(owner: Mapping[str, Any]) -> bool:
    if owner.get("hostname") != socket.gethostname():
        return False
    try:
        pid = int(owner.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if not _pid_is_alive(pid):
        return False
    recorded = owner.get("process_created_unix_ns")
    current = _process_created_unix_ns(pid)
    # Old or minimally provisioned lock records cannot be disproved safely.
    if recorded is None or current is None:
        return True
    try:
        return abs(int(recorded) - int(current)) <= 1_000_000
    except (TypeError, ValueError):
        return False


def _lock_heartbeat_age_seconds(
    owner: Mapping[str, Any], owner_path: Path, lock_dir: Path
) -> float:
    try:
        heartbeat_ns = int(owner["last_heartbeat_unix_ns"])
        if heartbeat_ns <= 0:
            raise ValueError("non-positive heartbeat")
        return max(0.0, (time.time_ns() - heartbeat_ns) / 1_000_000_000.0)
    except (KeyError, TypeError, ValueError):
        witness = owner_path if owner_path.exists() else lock_dir
        return max(0.0, time.time() - witness.stat().st_mtime)


@contextmanager
def _maintain_lock_heartbeat(
    *, lock_dir: Path, owner_path: Path, token: str
) -> Iterable[None]:
    """Refresh a crash-stopping lease without weakening exclusive ownership."""

    stop = threading.Event()
    failures: list[str] = []

    def refresh() -> None:
        sequence = 0
        while not stop.wait(LOCK_HEARTBEAT_SECONDS):
            try:
                current = _load_json(owner_path)
                if current.get("token") != token:
                    return
                sequence = int(current.get("heartbeat_sequence", sequence)) + 1
                current["heartbeat_sequence"] = sequence
                current["last_heartbeat_unix_ns"] = time.time_ns()
                current["last_heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(current, owner_path)
                try:
                    os.utime(lock_dir, None)
                except FileNotFoundError:
                    return
            except Exception as error:  # keep a forensic reason for the owner.
                failures.append(f"{type(error).__name__}: {error}")
                return

    worker = threading.Thread(
        target=refresh,
        name=f"t9_1_3_lock_heartbeat_{token[:12]}",
        daemon=True,
    )
    worker.start()
    body_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        body_error = error
        raise
    finally:
        stop.set()
        worker.join(timeout=max(2.0, LOCK_HEARTBEAT_SECONDS + 1.0))
        if worker.is_alive() and body_error is None:
            raise RuntimeError("lock heartbeat thread did not terminate")
        if failures and body_error is None:
            raise RuntimeError(f"lock heartbeat failed: {failures[-1]}")


@contextmanager
def _agent_writer_lock(
    *,
    output_dir: Path,
    strategy: str,
    seed: int,
    config: Mapping[str, Any],
    implementation_hash: str,
) -> Iterable[None]:
    """Provide a crash-detectable single writer for one strategy/root shard."""

    lock_dir = output_dir / "_locks" / f"{strategy}_seed_{seed}.lock"
    owner_path = lock_dir / "owner.json"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(
        f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}".encode("utf-8")
    ).hexdigest()
    runtime_signature, runtime_signature_hash = _current_training_runtime()
    metadata = {
        "schema_version": "t9.1.3-writer-lock-v1",
        "task_id": TASK_ID,
        "strategy": strategy,
        "root_seed": int(seed),
        "pid": os.getpid(),
        "process_created_unix_ns": _process_created_unix_ns(os.getpid()),
        "hostname": socket.gethostname(),
        "token": token,
        "agent_contract_hash": _agent_contract_hash(
            config, strategy, seed, implementation_hash, runtime_signature
        ),
        "training_runtime_signature_sha256": runtime_signature_hash,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat_unix_ns": time.time_ns(),
        "last_heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        "heartbeat_sequence": 0,
        "lease_seconds": LOCK_LEASE_SECONDS,
    }
    while True:
        try:
            lock_dir.mkdir(exist_ok=False)
        except FileExistsError:
            try:
                age_seconds = max(0.0, time.time() - lock_dir.stat().st_mtime)
            except FileNotFoundError:
                continue
            try:
                incumbent = _load_json(owner_path)
            except Exception as error:
                incumbent = {
                    "pid": -1,
                    "hostname": socket.gethostname(),
                    "parse_error": f"{type(error).__name__}: {error}",
                }
                if age_seconds < 60.0:
                    raise ConcurrentAgentWriterError(
                        f"writer lock for {strategy} seed {seed} is initializing"
                    ) from error
            age_seconds = _lock_heartbeat_age_seconds(
                incumbent, owner_path, lock_dir
            )
            if _local_owner_is_alive(incumbent):
                raise ConcurrentAgentWriterError(
                    f"live writer owns {strategy} seed {seed}: pid={incumbent.get('pid')}"
                )
            local_dead_owner = (
                incumbent.get("hostname") == socket.gethostname()
                and not _local_owner_is_alive(incumbent)
            )
            if not local_dead_owner:
                raise ConcurrentAgentWriterError(
                    f"cross-host writer lock for {strategy} seed {seed} is "
                    "fail-closed and requires explicit operator recovery; "
                    f"heartbeat_age_seconds={age_seconds:.1f}"
                )
            stale_dir = _new_attempt_directory(
                output_dir, "_failed_attempts", strategy, seed
            )
            stale_path = stale_dir / "stale_writer_lock"
            try:
                os.replace(lock_dir, stale_path)
            except FileNotFoundError:
                continue
            archived_lock_files = [
                _archive_file_entry(path)
                for path in sorted(stale_path.rglob("*"))
                if path.is_file()
            ]
            failure = _failure_record(
                config=config,
                implementation_hash=implementation_hash,
                strategy=strategy,
                seed=seed,
                failure_state="NULL",
                failure_subtype="STALE_WRITER_LOCK",
                exception_type="InterruptedWriter",
                exception_message="stale writer lock was retained before retry",
                traceback_text="",
                archived_partial_files=archived_lock_files,
                safe_retry_authorized=True,
            )
            failure["stale_lock_metadata"] = incumbent
            _atomic_json(failure, stale_dir / "failure.json")
            continue
        else:
            _atomic_json(metadata, owner_path)
            break
    try:
        with _maintain_lock_heartbeat(
            lock_dir=lock_dir, owner_path=owner_path, token=token
        ):
            yield
    finally:
        try:
            current = _load_json(owner_path)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
            current = None
        if current is not None and current.get("token") == token:
            owner_path.unlink()
            lock_dir.rmdir()


def _assert_finalize_not_active(output_dir: Path) -> None:
    if (output_dir / "_locks" / "finalize.lock").exists():
        raise ConcurrentAgentWriterError(
            "T9.1.3 finalization owns the output namespace"
        )


def _is_canonical_finalize_lock_management_file(
    path: Path, output_dir: Path
) -> bool:
    """Exclude only bytes owned by the active canonical finalization lock."""

    try:
        relative = path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        return False
    return (
        len(relative.parts) >= 3
        and relative.parts[0] == "_locks"
        and relative.parts[1] == "finalize.lock"
    )


def _seal_lock_history_recovery(
    directory: Path, stale_owner: Mapping[str, Any] | None = None
) -> None:
    recovery_path = directory / "recovery.json"
    if recovery_path.is_file():
        retained = directory / f"recovery_payload_retained_{time.time_ns()}.json"
        os.replace(recovery_path, retained)
    owner_path = directory / "owner.json"
    if stale_owner is None and owner_path.is_file():
        try:
            stale_owner = _load_json(owner_path)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            stale_owner = None
    archived_paths = sorted(
        path for path in directory.rglob("*") if path.is_file() and path != recovery_path
    )
    _atomic_json(
        {
            "schema_version": "t9.1.3-finalize-lock-recovery-v1",
            "task_id": TASK_ID,
            "stale_owner": dict(stale_owner) if stale_owner is not None else None,
            "owner_sha256": _file_sha256(owner_path) if owner_path.is_file() else None,
            "archived_files": [_archive_file_entry(path) for path in archived_paths],
            "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        recovery_path,
    )


def _recover_incomplete_lock_history(output_dir: Path) -> None:
    """Seal a finalize-lock archive interrupted between move and marker write."""

    root = output_dir / "_lock_history"
    if not root.exists():
        return
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        recovery_path = directory / "recovery.json"
        valid = False
        if recovery_path.is_file():
            try:
                recovery = _load_json(recovery_path)
                entries = recovery.get("archived_files")
                declared = {
                    _resolve_repo_path(str(entry["path"])).resolve()
                    for entry in entries
                }
                actual = {
                    path.resolve()
                    for path in directory.rglob("*")
                    if path.is_file() and path != recovery_path
                }
                valid = bool(
                    recovery.get("schema_version")
                    == "t9.1.3-finalize-lock-recovery-v1"
                    and recovery.get("task_id") == TASK_ID
                    and isinstance(entries, list)
                    and declared == actual
                    and all(
                        int(entry["bytes"])
                        == _resolve_repo_path(str(entry["path"])).stat().st_size
                        and entry["sha256"]
                        == _file_sha256(_resolve_repo_path(str(entry["path"])))
                        for entry in entries
                    )
                )
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                OSError,
            ):
                valid = False
        if not valid:
            _seal_lock_history_recovery(directory)


def _seal_finalization_recovery(
    directory: Path, original_paths: Sequence[str] = ()
) -> None:
    recovery_path = directory / "recovery.json"
    if recovery_path.is_file():
        retained = directory / f"recovery_payload_retained_{time.time_ns()}.json"
        os.replace(recovery_path, retained)
    archived = sorted(
        path for path in directory.rglob("*") if path.is_file() and path != recovery_path
    )
    _atomic_json(
        {
            "schema_version": "t9.1.3-finalization-atomic-recovery-v1",
            "task_id": TASK_ID,
            "original_paths": list(original_paths),
            "archived_files": [_archive_file_entry(path) for path in archived],
            "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        recovery_path,
    )


def _recover_incomplete_finalization_recoveries(output_dir: Path) -> None:
    root = output_dir / "_finalization_recovery"
    if not root.exists():
        return
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        recovery_path = directory / "recovery.json"
        valid = False
        if recovery_path.is_file():
            try:
                recovery = _load_json(recovery_path)
                entries = recovery["archived_files"]
                declared = {
                    _resolve_repo_path(str(entry["path"])).resolve()
                    for entry in entries
                }
                actual = {
                    path.resolve()
                    for path in directory.rglob("*")
                    if path.is_file() and path != recovery_path
                }
                valid = bool(
                    recovery.get("schema_version")
                    == "t9.1.3-finalization-atomic-recovery-v1"
                    and recovery.get("task_id") == TASK_ID
                    and isinstance(recovery.get("original_paths"), list)
                    and declared == actual
                    and all(
                        int(entry["bytes"])
                        == _resolve_repo_path(str(entry["path"])).stat().st_size
                        and entry["sha256"]
                        == _file_sha256(_resolve_repo_path(str(entry["path"])))
                        for entry in entries
                    )
                )
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                OSError,
            ):
                valid = False
        if not valid:
            _seal_finalization_recovery(directory)


def _recover_finalization_atomic_temporaries(
    output_dir: Path, targets: Sequence[Path]
) -> Path | None:
    """Retain hard-kill temporary ledgers before rebuilding final outputs."""

    candidates: list[Path] = []
    for target in targets:
        candidates.extend(
            path
            for path in target.parent.glob(f".{target.name}.*.tmp")
            if path.is_file()
        )
    if not candidates:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / "_finalization_recovery" / f"finalize_{stamp}_{time.time_ns()}"
    directory.mkdir(parents=True, exist_ok=False)
    original_paths: list[str] = []
    for index, source in enumerate(sorted(set(candidates))):
        original_paths.append(_portable_artifact_path(source))
        os.replace(source, directory / f"atomic_temp_{index:04d}_{source.name}")
    _seal_finalization_recovery(directory, original_paths)
    return directory / "recovery.json"


@contextmanager
def _namespace_finalize_lock(output_dir: Path) -> Iterable[None]:
    """Exclude writers while final ledgers and their filesystem census are sealed."""

    lock_dir = output_dir / "_locks" / "finalize.lock"
    owner_path = lock_dir / "owner.json"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(
        f"finalize:{socket.gethostname()}:{os.getpid()}:{time.time_ns()}".encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = {
        "schema_version": "t9.1.3-finalize-lock-v1",
        "task_id": TASK_ID,
        "pid": os.getpid(),
        "process_created_unix_ns": _process_created_unix_ns(os.getpid()),
        "hostname": socket.gethostname(),
        "token": token,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat_unix_ns": time.time_ns(),
        "last_heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        "heartbeat_sequence": 0,
        "lease_seconds": LOCK_LEASE_SECONDS,
    }
    while True:
        try:
            lock_dir.mkdir(exist_ok=False)
        except FileExistsError:
            try:
                age_seconds = max(0.0, time.time() - lock_dir.stat().st_mtime)
            except FileNotFoundError:
                continue
            try:
                incumbent = _load_json(owner_path)
            except Exception as error:
                if age_seconds < 60.0:
                    raise ConcurrentAgentWriterError(
                        "finalize lock is initializing"
                    ) from error
                incumbent = {
                    "pid": -1,
                    "hostname": socket.gethostname(),
                    "parse_error": f"{type(error).__name__}: {error}",
                }
            age_seconds = _lock_heartbeat_age_seconds(
                incumbent, owner_path, lock_dir
            )
            local_alive = _local_owner_is_alive(incumbent)
            local_dead = (
                incumbent.get("hostname") == socket.gethostname()
                and not local_alive
            )
            if local_alive:
                raise ConcurrentAgentWriterError("a live finalizer owns the namespace")
            owner_scope = "local dead" if local_dead else "cross-host or unverifiable"
            raise ConcurrentAgentWriterError(
                f"a {owner_scope} finalizer lock is fail-closed and requires "
                f"explicit operator recovery; heartbeat_age_seconds={age_seconds:.1f}"
            )
        else:
            _atomic_json(metadata, owner_path)
            break
    try:
        # A writer may have passed its first preflight check immediately before
        # this lock was created.  Existing per-agent locks expose that race.
        active_agents = [
            path
            for path in lock_dir.parent.glob("*_seed_*.lock")
            if path.is_dir()
        ]
        if active_agents:
            raise ConcurrentAgentWriterError(
                f"agent writers remain active during finalization: {active_agents}"
            )
        with _maintain_lock_heartbeat(
            lock_dir=lock_dir, owner_path=owner_path, token=token
        ):
            yield
    finally:
        try:
            current = _load_json(owner_path)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
            current = None
        if current is not None and current.get("token") == token:
            owner_path.unlink()
            lock_dir.rmdir()


def _recover_interrupted_training_attempts(
    *,
    output_dir: Path,
    strategy: str,
    seed: int,
    config: Mapping[str, Any],
    directional: DirectionalRankingConfig,
    implementation_hash: str,
) -> None:
    root = output_dir / "_training_attempts"
    if not root.exists():
        return
    runtime_signature, runtime_signature_hash = _current_training_runtime()
    pattern = f"{strategy}_seed_{seed}_*"
    for directory in sorted(path for path in root.glob(pattern) if path.is_dir()):
        start_path = directory / "start.json"
        outcome_path = directory / "outcome.json"
        failure_path = directory / "failure.json"
        if outcome_path.exists() or failure_path.exists():
            continue
        if not start_path.exists():
            recovered_attempt_id = _attempt_id(directory)
            _atomic_json(
                {
                    "schema_version": "t9.1.3-training-attempt-start-v1",
                    "task_id": TASK_ID,
                    "strategy": strategy,
                    "root_seed": int(seed),
                    "pid": -1,
                    "process_created_unix_ns": None,
                    "hostname": socket.gethostname(),
                    "config_sha256": _canonical_sha256(config),
                    "agent_contract_hash": None,
                    "training_runtime_signature": None,
                    "training_runtime_signature_sha256": None,
                    "runtime_signature_state": "UNKNOWN_START_COMMIT_INTERRUPTED",
                    "implementation_sha256": implementation_hash,
                    "parent_analysis_sha256": T9_1_1_ANALYSIS_SHA256,
                    "gpu_load_attestation_binding": None,
                    "attempt_id": recovered_attempt_id,
                    "state": "RECOVERED_START_MARKER_INTERRUPTED",
                    "started_at_utc": None,
                    "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                start_path,
            )
            failure = _failure_record(
                config=config,
                implementation_hash=implementation_hash,
                strategy=strategy,
                seed=seed,
                failure_state="NULL",
                failure_subtype="START_MARKER_INTERRUPTED",
                exception_type="InterruptedProcess",
                exception_message=(
                    "attempt directory existed without an authoritative start marker"
                ),
                traceback_text="",
                safe_retry_authorized=True,
            )
            failure["attempt_id"] = recovered_attempt_id
            failure["training_attempt_start_path"] = _portable_artifact_path(
                start_path
            )
            _atomic_json(failure, failure_path)
            continue
        start = _load_json(start_path)
        attempt_id = start.get("attempt_id")
        stored_signature = start.get("training_runtime_signature")
        if not isinstance(stored_signature, Mapping):
            raise AgentContractDriftError(
                f"training-attempt runtime signature missing for {strategy} seed {seed}"
            )
        try:
            _, stored_runtime_hash = _validated_training_runtime_signature(
                stored_signature,
                label=f"training attempt {strategy} seed {seed}",
            )
        except CorruptAgentArtifactError as error:
            raise AgentContractDriftError(str(error)) from error
        if (
            start.get("training_runtime_signature_sha256")
            != stored_runtime_hash
            or start.get("runtime_signature_state") != "CAPTURED_BEFORE_TRAINING"
            or start.get("agent_contract_hash")
            != _agent_contract_hash(
                config, strategy, seed, implementation_hash, stored_signature
            )
            or not _training_gpu_attestation_binding_valid(
                start.get("gpu_load_attestation_binding"), config
            )
            or not isinstance(attempt_id, str)
            or attempt_id != _attempt_id(directory)
        ):
            raise AgentContractDriftError(
                f"stale training attempt contract/identity drift for {strategy} seed {seed}"
            )
        if (
            start.get("hostname") == socket.gethostname()
            and _local_owner_is_alive(start)
            and int(start.get("pid", -1)) != os.getpid()
        ):
            raise ConcurrentAgentWriterError(
                f"live training attempt owns {strategy} seed {seed}"
            )
        loaded = None
        if stored_runtime_hash == runtime_signature_hash:
            try:
                loaded = _load_agent_record(
                    output_dir=output_dir,
                    strategy=strategy,
                    seed=seed,
                    directional=directional,
                    config=config,
                    implementation_hash=implementation_hash,
                )
            except AgentContractDriftError:
                raise
            except AgentArtifactError:
                loaded = None
        if loaded is not None and loaded[1].get("attempt_id") == attempt_id:
            record = loaded[1]
            _atomic_json(
                {
                    "schema_version": "t9.1.3-training-attempt-outcome-v1",
                    "task_id": TASK_ID,
                    "strategy": strategy,
                    "root_seed": int(seed),
                    "attempt_id": attempt_id,
                    "state": "SUCCESS",
                    "agent_contract_hash": record["agent_contract_hash"],
                    "training_runtime_signature_sha256": record[
                        "training_runtime_signature_sha256"
                    ],
                    "gpu_load_attestation_binding": record[
                        "gpu_load_attestation_binding"
                    ],
                    "canonical_tensor_sha256": record[
                        "canonical_tensor_sha256"
                    ],
                    "weight_file_sha256": record["weight_file_sha256"],
                    "terminal_file_sha256": record["terminal_file_sha256"],
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "recovered_after_interrupted_outcome_write": True,
                },
                outcome_path,
            )
            continue
        failure = _failure_record(
            config=config,
            implementation_hash=implementation_hash,
            strategy=strategy,
            seed=seed,
            failure_state="NULL",
            failure_subtype=(
                "INTERRUPTED_BEFORE_OUTCOME"
                if stored_runtime_hash == runtime_signature_hash
                else "INTERRUPTED_RUNTIME_CHANGED_BEFORE_OUTCOME"
            ),
            exception_type="InterruptedProcess",
            exception_message="STARTED attempt had no terminal outcome on resume",
            traceback_text="",
            safe_retry_authorized=True,
            runtime_signature=stored_signature,
        )
        failure["attempt_id"] = attempt_id
        failure["training_attempt_start_path"] = _portable_artifact_path(start_path)
        _atomic_json(failure, failure_path)


def _start_training_attempt(
    *,
    output_dir: Path,
    strategy: str,
    seed: int,
    config: Mapping[str, Any],
    implementation_hash: str,
    gpu_load_attestation_binding: Mapping[str, Any] | None,
) -> Path:
    directory = _new_attempt_directory(
        output_dir, "_training_attempts", strategy, seed
    )
    attempt_id = _attempt_id(directory)
    runtime_signature, runtime_signature_hash = _current_training_runtime()
    _atomic_json(
        {
            "schema_version": "t9.1.3-training-attempt-start-v1",
            "task_id": TASK_ID,
            "strategy": strategy,
            "root_seed": int(seed),
            "pid": os.getpid(),
            "process_created_unix_ns": _process_created_unix_ns(os.getpid()),
            "hostname": socket.gethostname(),
            "config_sha256": _canonical_sha256(config),
            "agent_contract_hash": _agent_contract_hash(
                config, strategy, seed, implementation_hash, runtime_signature
            ),
            "training_runtime_signature": runtime_signature,
            "training_runtime_signature_sha256": runtime_signature_hash,
            "runtime_signature_state": "CAPTURED_BEFORE_TRAINING",
            "implementation_sha256": implementation_hash,
            "parent_analysis_sha256": T9_1_1_ANALYSIS_SHA256,
            "gpu_load_attestation_binding": (
                dict(gpu_load_attestation_binding)
                if gpu_load_attestation_binding is not None
                else None
            ),
            "attempt_id": attempt_id,
            "state": "STARTED",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        directory / "start.json",
    )
    return directory


def _optimizer_state_cpu(value: Any) -> Any:
    th = _require_torch()
    if isinstance(value, th.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _optimizer_state_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_optimizer_state_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_state_cpu(item) for item in value)
    return value


TRAINING_METRIC_KEYS = (
    "mean_reward",
    "score_path",
    "loss",
    "gradient_norm_before_clip",
    "baseline",
    "ground_outcome_fraction",
    "control_residual_rms",
    "control_slew_rms",
)
TRAINING_LINEAGE_SCHEMA_VERSION = "t9.1.3-training-lineage-v1"
TRAINING_CURVE_ROW_KEYS = (
    "epoch",
    *TRAINING_METRIC_KEYS,
    "training_trajectory_seed",
    "lineage_strategy",
    "lineage_root_seed",
    "pre_model_sha256",
    "post_model_sha256",
    "pre_optimizer_sha256",
    "post_optimizer_sha256",
    "pre_rng_sha256",
    "post_rng_sha256",
    "previous_lineage_sha256",
    "epoch_metrics_sha256",
    "lineage_sha256",
)


def _training_state_sha256(value: Any) -> str:
    """Hash nested optimizer/RNG state without relying on pickle byte layout."""

    th = _require_torch()
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, th.Tensor):
            array = np.ascontiguousarray(item.detach().cpu().contiguous().numpy())
            digest.update(b"torch\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            ordered = sorted(
                item.items(),
                key=lambda pair: (type(pair[0]).__name__, repr(pair[0])),
            )
            digest.update(np.asarray([len(ordered)], dtype=np.int64).tobytes())
            for key, nested in ordered:
                update(key)
                update(nested)
            return
        if isinstance(item, tuple):
            digest.update(b"tuple\0")
            digest.update(np.asarray([len(item)], dtype=np.int64).tobytes())
            for nested in item:
                update(nested)
            return
        if isinstance(item, list):
            digest.update(b"list\0")
            digest.update(np.asarray([len(item)], dtype=np.int64).tobytes())
            for nested in item:
                update(nested)
            return
        if item is None:
            digest.update(b"none\0")
            return
        if isinstance(item, (bool, np.bool_)):
            digest.update(b"bool\0")
            digest.update(b"1" if bool(item) else b"0")
            return
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            digest.update(b"int\0")
            digest.update(str(int(item)).encode("ascii"))
            digest.update(b"\0")
            return
        if isinstance(item, (float, np.floating)):
            digest.update(b"float\0")
            digest.update(float(item).hex().encode("ascii"))
            digest.update(b"\0")
            return
        if isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"str\0")
            digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
            digest.update(encoded)
            return
        raise TypeError(f"unsupported training-state value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _training_rng_snapshot() -> dict[str, Any]:
    th = _require_torch()
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": th.get_rng_state().detach().cpu().clone(),
        "torch_cuda": (
            [state.detach().cpu().clone() for state in th.cuda.get_rng_state_all()]
            if th.cuda.is_available()
            else []
        ),
        "numpy": (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32).copy(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
    }


def _agent_local_rng_contract(strategy: str, seed: int) -> dict[str, Any]:
    if strategy not in {"mf", "nmf"} or int(seed) < 0:
        raise ValueError("agent-local RNG identity is invalid")
    digest = hashlib.sha256(
        (
            f"{TASK_ID}|agent-local-rng-v1|strategy={strategy}|root_seed={int(seed)}"
        ).encode("ascii")
    ).digest()
    return {
        "schema_version": "t9.1.3-agent-local-rng-v1",
        "domain_separation": f"{TASK_ID}:agent-local-rng-v1",
        "strategy": strategy,
        "root_seed": int(seed),
        "torch_cpu_cuda_seed": int.from_bytes(digest[:8], "big")
        & ((1 << 63) - 1),
        "numpy_legacy_mt19937_seed": int.from_bytes(digest[8:12], "big"),
        "initialized_before_model_construction": True,
    }


def _initialize_agent_local_rng(strategy: str, seed: int) -> dict[str, Any]:
    th = _require_torch()
    contract = _agent_local_rng_contract(strategy, seed)
    np.random.seed(int(contract["numpy_legacy_mt19937_seed"]))
    th.manual_seed(int(contract["torch_cpu_cuda_seed"]))
    if th.cuda.is_available():
        th.cuda.manual_seed_all(int(contract["torch_cpu_cuda_seed"]))
    return contract


def _validated_agent_local_rng_contract(
    value: Any, *, strategy: str, seed: int
) -> dict[str, Any]:
    expected = _agent_local_rng_contract(strategy, seed)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise CorruptAgentArtifactError(
            f"agent-local RNG contract drift for {strategy} seed {seed}"
        )
    return expected


def _terminal_rng_snapshot(terminal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "torch_cpu": terminal.get("torch_cpu_rng_state"),
        "torch_cuda": terminal.get("torch_cuda_rng_states"),
        "numpy": terminal.get("numpy_random_state"),
    }


def _training_lineage_genesis_sha256(
    *,
    strategy: str,
    seed: int,
    initial_model_sha256: str,
    initial_optimizer_sha256: str,
    initial_rng_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": TRAINING_LINEAGE_SCHEMA_VERSION,
            "strategy": strategy,
            "root_seed": int(seed),
            "epoch": 0,
            "initial_model_sha256": initial_model_sha256,
            "initial_optimizer_sha256": initial_optimizer_sha256,
            "initial_rng_sha256": initial_rng_sha256,
        }
    )


def _training_metric_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {name: float(row[name]) for name in TRAINING_METRIC_KEYS}
    )


def _training_epoch_lineage_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "schema_version": TRAINING_LINEAGE_SCHEMA_VERSION,
            "strategy": str(row["lineage_strategy"]),
            "root_seed": int(row["lineage_root_seed"]),
            "epoch": int(row["epoch"]),
            "training_trajectory_seed": int(row["training_trajectory_seed"]),
            "previous_lineage_sha256": row["previous_lineage_sha256"],
            "pre_model_sha256": row["pre_model_sha256"],
            "post_model_sha256": row["post_model_sha256"],
            "pre_optimizer_sha256": row["pre_optimizer_sha256"],
            "post_optimizer_sha256": row["post_optimizer_sha256"],
            "pre_rng_sha256": row["pre_rng_sha256"],
            "post_rng_sha256": row["post_rng_sha256"],
            "epoch_metrics_sha256": row["epoch_metrics_sha256"],
        }
    )


def _validation_final_fidelity(evaluation: Mapping[str, Any]) -> float:
    """Return the validation-only mean final-state fidelity used for selection."""

    values = np.asarray(
        [row["fidelity_curve"][-1] for row in evaluation["per_seed"]],
        dtype=np.float64,
    )
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("validation final fidelities must be nonempty and finite")
    return float(np.mean(values))


def _feedback_grape_loss(
    result: Any,
    *,
    baseline: float,
    nominal: Any,
    residual_l2_weight: float,
    slew_l2_weight: float,
    weights: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Shared sampled/exact-tree loss; regularizer score omission is deliberate D33."""

    th = _require_torch()
    residual_per_trajectory, slew_per_trajectory = (
        applied_control_penalties_per_trajectory(result, nominal)
    )
    if weights is None:
        normalized_weights = th.full_like(
            result.reward, 1.0 / float(result.reward.numel())
        )
    else:
        if not isinstance(weights, th.Tensor):
            raise TypeError("Feedback-GRAPE weights must be a torch.Tensor")
        normalized_weights = weights.to(
            device=result.reward.device, dtype=result.reward.dtype
        )
        if (
            normalized_weights.requires_grad
            or normalized_weights.shape != result.reward.shape
            or not bool(th.all(th.isfinite(normalized_weights)).detach().cpu())
            or bool(th.any(normalized_weights < 0.0).detach().cpu())
            or not bool(
                th.isclose(
                    normalized_weights.sum(),
                    th.ones((), dtype=result.reward.dtype, device=result.reward.device),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                .detach()
                .cpu()
            )
        ):
            raise ValueError("Feedback-GRAPE weights must be detached probabilities")
    reward_path = th.sum(normalized_weights * result.reward)
    score_path = th.sum(
        normalized_weights
        * (result.reward.detach() - float(baseline))
        * result.log_probability
    )
    residual_penalty = th.sum(normalized_weights * residual_per_trajectory)
    slew_penalty = th.sum(normalized_weights * slew_per_trajectory)
    loss = (
        -(reward_path + score_path)
        + float(residual_l2_weight) * residual_penalty
        + float(slew_l2_weight) * slew_penalty
    )
    return loss, reward_path, score_path, residual_penalty, slew_penalty


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return bool(np.isfinite(value))
    return False


def _assert_finite_state_dict(
    state: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    label: str,
) -> None:
    th = _require_torch()
    if not isinstance(state, Mapping) or set(state) != set(reference):
        raise CorruptAgentArtifactError(f"{label} tensor-name census drifted")
    for name, reference_tensor in reference.items():
        tensor = state.get(name)
        if (
            not isinstance(tensor, th.Tensor)
            or tensor.shape != reference_tensor.shape
            or tensor.dtype != reference_tensor.dtype
            or not bool(th.all(th.isfinite(tensor)).detach().cpu())
        ):
            raise CorruptAgentArtifactError(f"{label} tensor {name} is invalid")


def _assert_nested_replay_match(
    stored: Any,
    replayed: Any,
    *,
    label: str,
) -> float:
    """Compare a stored validation tree with a fresh deterministic replay."""

    if isinstance(stored, Mapping):
        if not isinstance(replayed, Mapping) or set(stored) != set(replayed):
            raise CorruptAgentArtifactError(f"{label} mapping schema drifted")
        return max(
            (
                _assert_nested_replay_match(
                    stored[key], replayed[key], label=f"{label}.{key}"
                )
                for key in stored
            ),
            default=0.0,
        )
    if isinstance(stored, (list, tuple)):
        if not isinstance(replayed, (list, tuple)) or len(stored) != len(replayed):
            raise CorruptAgentArtifactError(f"{label} sequence census drifted")
        return max(
            (
                _assert_nested_replay_match(
                    left, right, label=f"{label}[{index}]"
                )
                for index, (left, right) in enumerate(
                    zip(stored, replayed, strict=True)
                )
            ),
            default=0.0,
        )
    if isinstance(stored, bool) or stored is None or isinstance(stored, str):
        if type(stored) is not type(replayed) or stored != replayed:
            raise CorruptAgentArtifactError(f"{label} categorical value drifted")
        return 0.0
    if isinstance(stored, (int, np.integer)) and not isinstance(stored, bool):
        if (
            isinstance(replayed, bool)
            or not isinstance(replayed, (int, np.integer))
            or int(stored) != int(replayed)
        ):
            raise CorruptAgentArtifactError(f"{label} integer value drifted")
        return 0.0
    if isinstance(stored, (float, np.floating)):
        if not isinstance(replayed, (int, float, np.integer, np.floating)):
            raise CorruptAgentArtifactError(f"{label} numeric type drifted")
        left = float(stored)
        right = float(replayed)
        error = abs(left - right)
        if (
            not np.isfinite(left)
            or not np.isfinite(right)
            or not np.isclose(left, right, rtol=2.0e-10, atol=2.0e-12)
        ):
            raise CorruptAgentArtifactError(
                f"{label} is not reproducible from the candidate checkpoint"
            )
        return error
    raise CorruptAgentArtifactError(f"{label} contains an unsupported value type")


def _validate_training_lineage(
    *,
    training_record: Mapping[str, Any],
    terminal: Mapping[str, Any],
    model: Any,
    directional: DirectionalRankingConfig,
    strategy: str,
    seed: int,
) -> dict[str, Any]:
    """Recompute the epoch chain and bind validation/terminal checkpoints to it."""

    th = _require_torch()
    curve = training_record.get("training_curve")
    candidates = terminal.get("validation_candidate_states")
    history = training_record.get("validation_history")
    if (
        not isinstance(curve, list)
        or len(curve) != int(directional.train_epochs)
        or not isinstance(candidates, list)
        or not isinstance(history, list)
    ):
        raise CorruptAgentArtifactError(
            f"training lineage census missing for {strategy} seed {seed}"
        )
    if (
        training_record.get("training_lineage_schema_version")
        != TRAINING_LINEAGE_SCHEMA_VERSION
        or terminal.get("training_lineage_schema_version")
        != TRAINING_LINEAGE_SCHEMA_VERSION
        or training_record.get("training_lineage_complete") is not True
    ):
        raise CorruptAgentArtifactError(
            f"training lineage schema drift for {strategy} seed {seed}"
        )

    candidate_by_epoch: dict[int, Mapping[str, Any]] = {}
    history_by_epoch: dict[int, Mapping[str, Any]] = {}
    for candidate, validation_row in zip(candidates, history, strict=True):
        epoch = int(candidate.get("epoch", -1))
        if epoch in candidate_by_epoch or int(validation_row.get("epoch", -1)) != epoch:
            raise CorruptAgentArtifactError(
                f"training lineage validation epoch drift for {strategy} seed {seed}"
            )
        candidate_by_epoch[epoch] = candidate
        history_by_epoch[epoch] = validation_row
    if 0 not in candidate_by_epoch or int(directional.train_epochs) not in candidate_by_epoch:
        raise CorruptAgentArtifactError(
            f"training lineage endpoint candidate missing for {strategy} seed {seed}"
        )

    try:
        initial_candidate_hash = state_dict_sha256(
            candidate_by_epoch[0]["model_state"]
        )
        terminal_candidate_hash = state_dict_sha256(
            candidate_by_epoch[int(directional.train_epochs)]["model_state"]
        )
        terminal_model_hash = state_dict_sha256(terminal["terminal_model_state"])
    except (KeyError, TypeError, AttributeError) as error:
        raise CorruptAgentArtifactError(
            f"training lineage endpoint tensors invalid for {strategy} seed {seed}"
        ) from error
    initial_model_hash = training_record.get("initial_state_sha256")
    terminal_record_hash = training_record.get("terminal_state_sha256")
    if (
        not _is_lower_sha256(initial_model_hash)
        or initial_candidate_hash != initial_model_hash
        or candidate_by_epoch[0].get("state_sha256") != initial_model_hash
        or history_by_epoch[0].get("candidate_state_sha256") != initial_model_hash
        or not _is_lower_sha256(terminal_record_hash)
        or terminal_candidate_hash != terminal_record_hash
        or terminal_model_hash != terminal_record_hash
        or candidate_by_epoch[int(directional.train_epochs)].get("state_sha256")
        != terminal_record_hash
        or history_by_epoch[int(directional.train_epochs)].get(
            "candidate_state_sha256"
        )
        != terminal_record_hash
    ):
        raise CorruptAgentArtifactError(
            f"training lineage endpoint binding failed for {strategy} seed {seed}"
        )

    first = curve[0]
    expected_empty_optimizer_hash = _training_state_sha256(
        _optimizer_state_cpu(
            th.optim.Adam(model.parameters(), lr=directional.learning_rate).state_dict()
        )
    )
    if first.get("pre_optimizer_sha256") != expected_empty_optimizer_hash:
        raise CorruptAgentArtifactError(
            f"training lineage initial optimizer drift for {strategy} seed {seed}"
        )
    genesis = _training_lineage_genesis_sha256(
        strategy=strategy,
        seed=seed,
        initial_model_sha256=initial_model_hash,
        initial_optimizer_sha256=first.get("pre_optimizer_sha256"),
        initial_rng_sha256=first.get("pre_rng_sha256"),
    )
    if (
        training_record.get("training_lineage_genesis_sha256") != genesis
        or terminal.get("training_lineage_genesis_sha256") != genesis
        or candidate_by_epoch[0].get("lineage_sha256") != genesis
        or history_by_epoch[0].get("training_lineage_sha256") != genesis
    ):
        raise CorruptAgentArtifactError(
            f"training lineage genesis mismatch for {strategy} seed {seed}"
        )

    previous_lineage = genesis
    previous_model = initial_model_hash
    previous_optimizer = first.get("pre_optimizer_sha256")
    previous_rng = first.get("pre_rng_sha256")
    chain_rows: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(curve, start=1):
        if set(row) != set(TRAINING_CURVE_ROW_KEYS):
            raise CorruptAgentArtifactError(
                f"training lineage row schema drift for {strategy} seed {seed}"
            )
        hashes = [
            row[name]
            for name in (
                "pre_model_sha256",
                "post_model_sha256",
                "pre_optimizer_sha256",
                "post_optimizer_sha256",
                "pre_rng_sha256",
                "post_rng_sha256",
                "previous_lineage_sha256",
                "epoch_metrics_sha256",
                "lineage_sha256",
            )
        ]
        if (
            int(row["epoch"]) != expected_epoch
            or row["lineage_strategy"] != strategy
            or int(row["lineage_root_seed"]) != int(seed)
            or int(row["training_trajectory_seed"])
            != _training_trajectory_seed(seed, expected_epoch)
            or not all(_is_lower_sha256(value) for value in hashes)
            or not _all_numeric_finite(
                {name: row[name] for name in TRAINING_METRIC_KEYS}
            )
            or float(row["gradient_norm_before_clip"]) <= 0.0
            or row["pre_model_sha256"] != previous_model
            or row["pre_optimizer_sha256"] != previous_optimizer
            or row["pre_rng_sha256"] != previous_rng
            or row["previous_lineage_sha256"] != previous_lineage
            or row["epoch_metrics_sha256"] != _training_metric_sha256(row)
            or row["lineage_sha256"] != _training_epoch_lineage_sha256(row)
        ):
            raise CorruptAgentArtifactError(
                f"training lineage chain failed at epoch {expected_epoch} for "
                f"{strategy} seed {seed}"
            )
        previous_model = row["post_model_sha256"]
        previous_optimizer = row["post_optimizer_sha256"]
        previous_rng = row["post_rng_sha256"]
        previous_lineage = row["lineage_sha256"]
        chain_rows.append(
            {
                "epoch": expected_epoch,
                "lineage_sha256": previous_lineage,
                "post_model_sha256": previous_model,
                "post_optimizer_sha256": previous_optimizer,
                "post_rng_sha256": previous_rng,
            }
        )
        if expected_epoch in candidate_by_epoch and (
            candidate_by_epoch[expected_epoch].get("lineage_sha256")
            != previous_lineage
            or history_by_epoch[expected_epoch].get("training_lineage_sha256")
            != previous_lineage
            or candidate_by_epoch[expected_epoch].get("state_sha256")
            != previous_model
            or history_by_epoch[expected_epoch].get("candidate_state_sha256")
            != previous_model
        ):
            raise CorruptAgentArtifactError(
                f"validation candidate is not the epoch post-state at epoch "
                f"{expected_epoch} for {strategy} seed {seed}"
            )

    terminal_optimizer_hash = _training_state_sha256(
        _optimizer_state_cpu(terminal.get("optimizer_state"))
    )
    terminal_rng_hash = _training_state_sha256(_terminal_rng_snapshot(terminal))
    if (
        previous_model != terminal_model_hash
        or previous_optimizer != terminal_optimizer_hash
        or previous_rng != terminal_rng_hash
        or training_record.get("training_lineage_terminal_sha256")
        != previous_lineage
        or terminal.get("training_lineage_terminal_sha256") != previous_lineage
    ):
        raise CorruptAgentArtifactError(
            f"training lineage terminal binding failed for {strategy} seed {seed}"
        )
    return {
        "schema_version": TRAINING_LINEAGE_SCHEMA_VERSION,
        "strategy": strategy,
        "root_seed": int(seed),
        "epoch_count": len(curve),
        "validation_candidate_count": len(candidates),
        "genesis_sha256": genesis,
        "terminal_lineage_sha256": previous_lineage,
        "chain_ledger_sha256": _canonical_sha256(chain_rows),
        "all_epochs_closed": True,
        "endpoints_bound": True,
        "validation_candidates_bound_to_epoch_post_state": True,
    }


def _validate_validation_evidence(
    *,
    training_record: Mapping[str, Any],
    terminal: Mapping[str, Any],
    model: Any,
    directional: DirectionalRankingConfig,
    strategy: str,
    seed: int,
    replay_physics: bool = False,
) -> dict[str, Any] | None:
    """Rebuild validation selection from raw values and reload every candidate."""

    history = training_record.get("validation_history")
    candidates = terminal.get("validation_candidate_states")
    if not isinstance(history, list) or not isinstance(candidates, list):
        raise CorruptAgentArtifactError(
            f"validation evidence missing for {strategy} seed {seed}"
        )
    expected_epochs = [0] + list(
        range(
            int(directional.validation_interval),
            int(directional.train_epochs) + 1,
            int(directional.validation_interval),
        )
    )
    if expected_epochs[-1] != int(directional.train_epochs):
        expected_epochs.append(int(directional.train_epochs))
    history_epochs = [int(row.get("epoch", -1)) for row in history]
    candidate_epochs = [int(row.get("epoch", -1)) for row in candidates]
    if history_epochs != expected_epochs or candidate_epochs != expected_epochs:
        raise CorruptAgentArtifactError(
            f"validation epoch census mismatch for {strategy} seed {seed}"
        )
    reference = model.state_dict()
    running_best = -np.inf
    independently_selected: list[int] = []
    replay_digests: list[dict[str, Any]] = []
    maximum_replay_error = 0.0
    for history_row, candidate_row in zip(history, candidates, strict=True):
        if set(history_row) != {
            "epoch",
            "selection_score",
            "metric_means",
            "validation_raw_per_seed",
            "candidate_state_sha256",
            "training_lineage_sha256",
            "selected_when_observed",
            "selected_final",
        }:
            raise CorruptAgentArtifactError(
                f"validation history schema drift for {strategy} seed {seed}"
            )
        if set(candidate_row) != {
            "epoch",
            "model_state",
            "state_sha256",
            "lineage_sha256",
        }:
            raise CorruptAgentArtifactError(
                f"validation candidate schema drift for {strategy} seed {seed}"
            )
        epoch = int(history_row["epoch"])
        score = float(history_row["selection_score"])
        if not np.isfinite(score) or not _all_numeric_finite(
            history_row.get("metric_means")
        ):
            raise CorruptAgentArtifactError(
                f"validation metric is non-finite for {strategy} seed {seed}"
            )
        per_seed = history_row.get("validation_raw_per_seed")
        if not isinstance(per_seed, list) or [
            int(row.get("seed", -1)) for row in per_seed
        ] != list(directional.validation_seeds):
            raise CorruptAgentArtifactError(
                f"validation seed ledger drift for {strategy} seed {seed}"
            )
        for raw in per_seed:
            if set(raw) != {
                "seed",
                "trajectory_count",
                "fidelity_curve",
                "trajectory_final_fidelity",
                "code_survival_curve",
                "logical_z_curve",
                "fidelity",
                "logical_z",
                "mean_ground_outcome_probability",
                "mean_control_residual_rms",
                "mean_control_slew_rms",
                "maximum_trace_error",
                "maximum_hermiticity_error",
                "minimum_final_eigenvalue",
            }:
                raise CorruptAgentArtifactError(
                    f"validation raw schema drift for {strategy} seed {seed}"
                )
            trajectory_values = np.asarray(
                raw.get("trajectory_final_fidelity"), dtype=np.float64
            )
            fidelity_curve = np.asarray(raw.get("fidelity_curve"), dtype=np.float64)
            survival_curve = np.asarray(
                raw.get("code_survival_curve"), dtype=np.float64
            )
            logical_z_curve = np.asarray(raw.get("logical_z_curve"), dtype=np.float64)
            lifetime_keys = {
                "effective_lifetime_cycles",
                "effective_lifetime_method",
                "normalized_auc",
                "log_linear_lifetime_cycles_diagnostic",
                "log_linear_fit_r_squared_diagnostic",
                "exponential_fit_is_diagnostic_only",
                "initial_value",
                "final_value",
            }
            lifetime_valid = True
            for lifetime in (raw.get("fidelity"), raw.get("logical_z")):
                if (
                    not isinstance(lifetime, Mapping)
                    or set(lifetime) != lifetime_keys
                    or lifetime.get("effective_lifetime_method")
                    != "finite_horizon_area_equivalent_exponential"
                    or lifetime.get("exponential_fit_is_diagnostic_only") is not True
                    or not all(
                        np.isfinite(float(lifetime[name]))
                        for name in lifetime_keys
                        - {
                            "effective_lifetime_method",
                            "exponential_fit_is_diagnostic_only",
                        }
                    )
                ):
                    lifetime_valid = False
                    break
            scalar_names = (
                "mean_ground_outcome_probability",
                "mean_control_residual_rms",
                "mean_control_slew_rms",
                "maximum_trace_error",
                "maximum_hermiticity_error",
                "minimum_final_eigenvalue",
            )
            if (
                not lifetime_valid
                or not all(np.isfinite(float(raw[name])) for name in scalar_names)
                or int(raw.get("trajectory_count", -1))
                != int(directional.validation_batch_size)
                or trajectory_values.shape
                != (int(directional.validation_batch_size),)
                or fidelity_curve.shape != (int(directional.full_cycles) + 1,)
                or survival_curve.shape != fidelity_curve.shape
                or logical_z_curve.shape != fidelity_curve.shape
                or not np.all(np.isfinite(fidelity_curve))
                or not np.all(np.isfinite(survival_curve))
                or not np.all(np.isfinite(logical_z_curve))
                or np.any(trajectory_values < -1.0e-9)
                or np.any(trajectory_values > 1.0 + 1.0e-9)
                or not np.isclose(
                    float(np.mean(trajectory_values)),
                    float(fidelity_curve[-1]),
                    rtol=0.0,
                    atol=2.0e-12,
                )
            ):
                raise CorruptAgentArtifactError(
                    f"validation raw trajectory ledger drift for {strategy} seed {seed}"
                )
        rebuilt_score = _validation_final_fidelity({"per_seed": per_seed})
        if not np.isclose(score, rebuilt_score, rtol=0.0, atol=0.0):
            raise CorruptAgentArtifactError(
                f"validation score is not raw-derived for {strategy} seed {seed}"
            )
        candidate_hash = history_row.get("candidate_state_sha256")
        if (
            not _is_lower_sha256(candidate_hash)
            or candidate_row.get("state_sha256") != candidate_hash
            or not _is_lower_sha256(history_row.get("training_lineage_sha256"))
            or candidate_row.get("lineage_sha256")
            != history_row.get("training_lineage_sha256")
        ):
            raise CorruptAgentArtifactError(
                f"validation candidate hash drift for {strategy} seed {seed}"
            )
        candidate_state = candidate_row.get("model_state")
        _assert_finite_state_dict(
            candidate_state,
            reference,
            label=f"validation candidate epoch {epoch} for {strategy} seed {seed}",
        )
        if state_dict_sha256(candidate_state) != candidate_hash:
            raise CorruptAgentArtifactError(
                f"validation candidate tensor hash mismatch for {strategy} seed {seed}"
            )
        try:
            model.load_state_dict(candidate_state, strict=True)
        except RuntimeError as error:
            raise CorruptAgentArtifactError(
                f"validation candidate cannot reload for {strategy} seed {seed}"
            ) from error
        if replay_physics:
            replayed = evaluate_policy(
                strategy,
                model,
                directional,
                cutoff=directional.cutoff,
                batch_size=directional.validation_batch_size,
                seeds=directional.validation_seeds,
            )
            maximum_replay_error = max(
                maximum_replay_error,
                _assert_nested_replay_match(
                    history_row["validation_raw_per_seed"],
                    replayed["per_seed"],
                    label=(
                        f"validation replay {strategy} seed {seed} epoch {epoch}.per_seed"
                    ),
                ),
                _assert_nested_replay_match(
                    history_row["metric_means"],
                    replayed["metric_means"],
                    label=(
                        f"validation replay {strategy} seed {seed} epoch {epoch}.metric_means"
                    ),
                ),
            )
            replayed_score = _validation_final_fidelity(replayed)
            maximum_replay_error = max(
                maximum_replay_error, abs(replayed_score - score)
            )
            if not np.isclose(
                replayed_score, score, rtol=2.0e-10, atol=2.0e-12
            ):
                raise CorruptAgentArtifactError(
                    f"validation score is not physics-reproducible for {strategy} "
                    f"seed {seed} epoch {epoch}"
                )
            replay_digests.append(
                {
                    "epoch": epoch,
                    "candidate_state_sha256": candidate_hash,
                    "replayed_evidence_sha256": _canonical_sha256(
                        {
                            "per_seed": replayed["per_seed"],
                            "metric_means": replayed["metric_means"],
                            "selection_score": replayed_score,
                        }
                    ),
                }
            )
        became_incumbent = score > running_best
        if history_row.get("selected_when_observed") is not became_incumbent:
            raise CorruptAgentArtifactError(
                f"validation incumbent flags drifted for {strategy} seed {seed}"
            )
        if became_incumbent:
            running_best = score
            independently_selected.append(epoch)
    best_row = sorted(history, key=lambda row: (-float(row["selection_score"]), int(row["epoch"])))[0]
    best_epoch = int(best_row["epoch"])
    if (
        int(training_record.get("best_validation_epoch", -1)) != best_epoch
        or not np.isclose(
            float(training_record.get("best_validation_score", np.nan)),
            float(best_row["selection_score"]),
            rtol=0.0,
            atol=0.0,
        )
        or sum(bool(row["selected_final"]) for row in history) != 1
        or any(
            bool(row["selected_final"]) != (int(row["epoch"]) == best_epoch)
            for row in history
        )
        or not independently_selected
    ):
        raise CorruptAgentArtifactError(
            f"validation best/tie contract drifted for {strategy} seed {seed}"
        )
    model.load_state_dict(terminal["validation_selected_state"], strict=True)
    if not replay_physics:
        return None
    return {
        "schema_version": "t9.1.3-validation-candidate-physics-replay-v1",
        "strategy": strategy,
        "root_seed": int(seed),
        "candidate_count": len(candidates),
        "validation_seed_count": len(directional.validation_seeds),
        "trajectories_per_candidate": (
            len(directional.validation_seeds)
            * int(directional.validation_batch_size)
        ),
        "maximum_numeric_absolute_error": float(maximum_replay_error),
        "candidate_replay_ledger_sha256": _canonical_sha256(replay_digests),
        "all_candidates_replayed": True,
    }


def _validate_optimizer_restart(
    optimizer_state: Any,
    model: Any,
    *,
    expected_epochs: int,
    expected_learning_rate: float,
    strategy: str,
    seed: int,
) -> None:
    th = _require_torch()
    if not isinstance(optimizer_state, Mapping):
        raise CorruptAgentArtifactError(
            f"optimizer state missing for {strategy} seed {seed}"
        )
    states = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    parameters = list(model.parameters())
    if (
        not isinstance(states, Mapping)
        or not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], Mapping)
        or not isinstance(groups[0].get("params"), list)
    ):
        raise CorruptAgentArtifactError(
            f"optimizer census malformed for {strategy} seed {seed}"
        )
    identifiers = list(groups[0]["params"])
    group = groups[0]
    if (
        len(identifiers) != len(parameters)
        or len(set(identifiers)) != len(parameters)
        or set(states) != set(identifiers)
        or set(group)
        != {
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
            "params",
        }
        or float(group["lr"]) != float(expected_learning_rate)
        or tuple(group["betas"]) != (0.9, 0.999)
        or float(group["eps"]) != 1.0e-8
        or float(group["weight_decay"]) != 0.0
        or any(
            group[name] is not False
            for name in (
                "amsgrad",
                "maximize",
                "capturable",
                "differentiable",
                "decoupled_weight_decay",
            )
        )
        or group["foreach"] is not None
        or group["fused"] is not None
    ):
        raise CorruptAgentArtifactError(
            f"optimizer parameter census drift for {strategy} seed {seed}"
        )
    for identifier, parameter in zip(identifiers, parameters, strict=True):
        row = states.get(identifier)
        if not isinstance(row, Mapping) or set(row) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise CorruptAgentArtifactError(
                f"optimizer slot schema drift for {strategy} seed {seed}"
            )
        step = row["step"]
        step_value = (
            float(step.detach().cpu())
            if isinstance(step, th.Tensor) and step.numel() == 1
            else float(step)
            if isinstance(step, (int, float)) and not isinstance(step, bool)
            else np.nan
        )
        if not np.isfinite(step_value) or step_value != float(expected_epochs):
            raise CorruptAgentArtifactError(
                f"optimizer step drift for {strategy} seed {seed}"
            )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = row[moment_name]
            if (
                not isinstance(moment, th.Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
                or not bool(th.all(th.isfinite(moment)).detach().cpu())
            ):
                raise CorruptAgentArtifactError(
                    f"optimizer {moment_name} invalid for {strategy} seed {seed}"
                )


def _train_paper_agent(
    strategy: Literal["mf", "nmf"],
    seed: int,
    config: DirectionalRankingConfig,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Train one agent while retaining selected and terminal restart state.

    This is intentionally isolated from the historical T2.3.7 runner.  It uses
    one persistent Adam optimizer, a train-only score baseline and validation-
    only checkpoint selection, and returns the terminal optimizer/RNG state as
    a recovery artifact.  Evaluation data never enters either selection level.
    """

    th = _require_torch()
    agent_local_rng_contract = _initialize_agent_local_rng(strategy, seed)
    model = build_policy(strategy, config, seed)
    model.train()
    initial_state = _state_dict_cpu(model)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    simulator = _simulator(config, cutoff=config.cutoff, batch_size=config.train_batch_size)
    nominal = nominal_sbs_parameters(
        device=config.device, dtype=_torch_dtype(config.real_dtype)
    )
    initial_validation = evaluate_policy(
        strategy,
        model,
        config,
        cutoff=config.cutoff,
        batch_size=config.validation_batch_size,
        seeds=config.validation_seeds,
    )
    best_score = _validation_final_fidelity(initial_validation)
    best_epoch = 0
    best_state = _state_dict_cpu(model)
    initial_hash = state_dict_sha256(best_state)
    warmup_seed = _training_trajectory_seed(seed, 0)
    with th.no_grad():
        warmup = simulator.run(
            control_policy=model, seed=warmup_seed, record_cycle_metrics=False
        )
    baseline = float(warmup.reward.detach().mean().cpu())
    initial_optimizer_hash = _training_state_sha256(
        _optimizer_state_cpu(optimizer.state_dict())
    )
    initial_rng_hash = _training_state_sha256(_training_rng_snapshot())
    lineage_genesis = _training_lineage_genesis_sha256(
        strategy=strategy,
        seed=seed,
        initial_model_sha256=initial_hash,
        initial_optimizer_sha256=initial_optimizer_hash,
        initial_rng_sha256=initial_rng_hash,
    )
    validation_candidate_states: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "model_state": best_state,
            "state_sha256": initial_hash,
            "lineage_sha256": lineage_genesis,
        }
    ]
    validation_history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "selection_score": best_score,
            "metric_means": initial_validation["metric_means"],
            "validation_raw_per_seed": initial_validation["per_seed"],
            "candidate_state_sha256": initial_hash,
            "training_lineage_sha256": lineage_genesis,
            "selected_when_observed": True,
            "selected_final": False,
        }
    ]
    training_curve: list[dict[str, Any]] = []
    previous_lineage_hash = lineage_genesis
    started = time.perf_counter()
    for epoch in range(1, config.train_epochs + 1):
        pre_model_hash = state_dict_sha256(_state_dict_cpu(model))
        pre_optimizer_hash = _training_state_sha256(
            _optimizer_state_cpu(optimizer.state_dict())
        )
        pre_rng_hash = _training_state_sha256(_training_rng_snapshot())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = simulator.run(
            control_policy=model,
            seed=_training_trajectory_seed(seed, epoch),
            record_cycle_metrics=False,
        )
        (
            loss,
            reward_path,
            score_path,
            residual_penalty,
            slew_penalty,
        ) = _feedback_grape_loss(
            result,
            baseline=baseline,
            nominal=nominal,
            residual_l2_weight=config.residual_l2_weight,
            slew_l2_weight=config.slew_l2_weight,
        )
        if not bool(th.isfinite(loss).detach().cpu()):
            raise RuntimeError(
                f"non-finite loss for {strategy} seed {seed} epoch {epoch}"
            )
        loss.backward()
        gradient_norm = float(
            th.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            .detach()
            .cpu()
        )
        if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(
                f"invalid gradient for {strategy} seed {seed} epoch {epoch}"
            )
        optimizer.step()
        reward_mean = float(result.reward.detach().mean().cpu())
        baseline = config.score_baseline_decay * baseline + (
            1.0 - config.score_baseline_decay
        ) * reward_mean
        training_row: dict[str, Any] = {
            "epoch": epoch,
            "mean_reward": reward_mean,
            "score_path": float(score_path.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": gradient_norm,
            "baseline": baseline,
            "ground_outcome_fraction": float(
                th.mean((result.outcomes == 0).to(th.float64)).detach().cpu()
            ),
            "control_residual_rms": float(th.sqrt(residual_penalty).detach().cpu()),
            "control_slew_rms": float(th.sqrt(slew_penalty).detach().cpu()),
        }
        validated_this_epoch = (
            epoch % config.validation_interval == 0 or epoch == config.train_epochs
        )
        if validated_this_epoch:
            validation = evaluate_policy(
                strategy,
                model,
                config,
                cutoff=config.cutoff,
                batch_size=config.validation_batch_size,
                seeds=config.validation_seeds,
            )
            score = _validation_final_fidelity(validation)
            candidate_state = _state_dict_cpu(model)
            candidate_hash = state_dict_sha256(candidate_state)
            improved = score > best_score
            validation_candidate_states.append(
                {
                    "epoch": int(epoch),
                    "model_state": candidate_state,
                    "state_sha256": candidate_hash,
                    "lineage_sha256": None,
                }
            )
            validation_history.append(
                {
                    "epoch": epoch,
                    "selection_score": score,
                    "metric_means": validation["metric_means"],
                    "validation_raw_per_seed": validation["per_seed"],
                    "candidate_state_sha256": candidate_hash,
                    "training_lineage_sha256": None,
                    "selected_when_observed": improved,
                    "selected_final": False,
                }
            )
            print(
                _canonical_json(
                    {
                        "event": "t9_1_3_validation",
                        "strategy": strategy,
                        "seed": seed,
                        "epoch": epoch,
                        "selection_score": score,
                        "improved": improved,
                    }
                ),
                flush=True,
            )
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = candidate_state
        post_model_hash = state_dict_sha256(_state_dict_cpu(model))
        post_optimizer_hash = _training_state_sha256(
            _optimizer_state_cpu(optimizer.state_dict())
        )
        post_rng_hash = _training_state_sha256(_training_rng_snapshot())
        training_row.update(
            {
                "training_trajectory_seed": _training_trajectory_seed(seed, epoch),
                "lineage_strategy": strategy,
                "lineage_root_seed": int(seed),
                "pre_model_sha256": pre_model_hash,
                "post_model_sha256": post_model_hash,
                "pre_optimizer_sha256": pre_optimizer_hash,
                "post_optimizer_sha256": post_optimizer_hash,
                "pre_rng_sha256": pre_rng_hash,
                "post_rng_sha256": post_rng_hash,
                "previous_lineage_sha256": previous_lineage_hash,
            }
        )
        training_row["epoch_metrics_sha256"] = _training_metric_sha256(training_row)
        training_row["lineage_sha256"] = _training_epoch_lineage_sha256(
            training_row
        )
        training_curve.append(training_row)
        if validated_this_epoch:
            if candidate_hash != post_model_hash:
                raise RuntimeError(
                    f"validation candidate/post-state mismatch for {strategy} "
                    f"seed {seed} epoch {epoch}"
                )
            validation_candidate_states[-1]["lineage_sha256"] = training_row[
                "lineage_sha256"
            ]
            validation_history[-1]["training_lineage_sha256"] = training_row[
                "lineage_sha256"
            ]
        previous_lineage_hash = training_row["lineage_sha256"]
    for row in validation_history:
        row["selected_final"] = int(row["epoch"]) == int(best_epoch)
    terminal_state = _state_dict_cpu(model)
    terminal_hash = state_dict_sha256(terminal_state)
    terminal_payload = {
        "schema_version": "t9.1.3-terminal-restart-v1",
        "strategy": strategy,
        "root_seed": int(seed),
        "terminal_epoch": int(config.train_epochs),
        "terminal_model_state": terminal_state,
        "terminal_model_sha256": terminal_hash,
        "validation_selected_state": best_state,
        "validation_selected_sha256": state_dict_sha256(best_state),
        "validation_candidate_states": validation_candidate_states,
        "optimizer_state": _optimizer_state_cpu(optimizer.state_dict()),
        "score_baseline": baseline,
        "torch_cpu_rng_state": th.get_rng_state().detach().cpu(),
        "torch_cuda_rng_states": (
            [item.detach().cpu() for item in th.cuda.get_rng_state_all()]
            if th.cuda.is_available()
            else []
        ),
        "numpy_random_state": np.random.get_state(),
        "training_lineage_schema_version": TRAINING_LINEAGE_SCHEMA_VERSION,
        "training_lineage_genesis_sha256": lineage_genesis,
        "training_lineage_terminal_sha256": previous_lineage_hash,
        "agent_local_rng_contract": agent_local_rng_contract,
    }
    model.load_state_dict(best_state)
    model.eval()
    record = {
        "strategy": strategy,
        "training_seed": int(seed),
        "parameter_count": int(model.parameter_count),
        "architecture": PAPER_MF_ARCHITECTURE
        if strategy == "mf"
        else PAPER_RNN_ARCHITECTURE,
        "epochs_executed": int(config.train_epochs),
        "training_trajectories": int(config.train_epochs * config.train_batch_size),
        "baseline_warmup_trajectories": int(config.train_batch_size),
        "baseline_initialization": "train-only no-gradient warm-up reward mean",
        "baseline_warmup_seed": warmup_seed,
        "training_trajectory_seeds": [
            _training_trajectory_seed(seed, epoch)
            for epoch in range(1, config.train_epochs + 1)
        ],
        "validation_seeds_used_for_checkpoint_selection_only": list(
            config.validation_seeds
        ),
        "best_validation_epoch": int(best_epoch),
        "best_validation_score": best_score,
        "initial_validation_score": float(
            _validation_final_fidelity(initial_validation)
        ),
        "validation_history": validation_history,
        "training_curve": training_curve,
        "checkpoint_sha256": state_dict_sha256(best_state),
        "initial_state_sha256": state_dict_sha256(initial_state),
        "terminal_state_sha256": terminal_hash,
        "training_lineage_schema_version": TRAINING_LINEAGE_SCHEMA_VERSION,
        "training_lineage_genesis_sha256": lineage_genesis,
        "training_lineage_terminal_sha256": previous_lineage_hash,
        "training_lineage_complete": True,
        "agent_local_rng_contract": agent_local_rng_contract,
        "continuous_adam_optimizer": True,
        "terminal_optimizer_state_retained": True,
        "terminal_rng_state_retained": True,
        "all_validation_candidate_states_retained": True,
        "validation_raw_per_seed_and_trajectory_retained": True,
        "wall_time_seconds": time.perf_counter() - started,
    }
    return model, record, terminal_payload


def _load_agent_record(
    *,
    output_dir: Path,
    strategy: Literal["mf", "nmf"],
    seed: int,
    directional: DirectionalRankingConfig,
    config: Mapping[str, Any],
    implementation_hash: str,
    replay_validation_candidates: bool = False,
) -> tuple[Any, dict[str, Any]] | None:
    th = _require_torch()
    weight_path, terminal_path, record_path = _agent_paths(
        output_dir, strategy, seed
    )
    if not weight_path.exists() and not terminal_path.exists() and not record_path.exists():
        return None
    if not weight_path.exists() or not terminal_path.exists() or not record_path.exists():
        raise PartialAgentArtifactError(
            f"partial/corrupt agent artifact for {strategy} seed {seed}"
        )
    try:
        record = _load_json(record_path)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise CorruptAgentArtifactError(
            f"agent record cannot be decoded for {strategy} seed {seed}"
        ) from error
    required_record_fields = {
        "schema_version",
        "task_id",
        "evidence_grade",
        "strategy",
        "root_seed",
        "config_sha256",
        "agent_contract_hash",
        "training_runtime_signature",
        "training_runtime_signature_sha256",
        "implementation_sha256",
        "parent_analysis_sha256",
        "gpu_load_attestation",
        "gpu_load_attestation_binding",
        "weight_file_sha256",
        "terminal_file_sha256",
        "canonical_tensor_sha256",
        "attempt_id",
        "strategy_config",
        "training_record",
        "weight_path",
        "weight_bytes",
        "terminal_path",
        "terminal_bytes",
        "prior_failure_record",
        "reload_witness",
        "terminal_reload_witness",
        "validation_candidate_reload_witness",
        "training_lineage_reload_witness",
    }
    if set(record) != required_record_fields or not isinstance(
        record.get("training_record"), dict
    ):
        raise CorruptAgentArtifactError(
            f"agent record schema incomplete for {strategy} seed {seed}"
        )
    stored_signature = record.get("training_runtime_signature")
    if not isinstance(stored_signature, Mapping):
        raise CorruptAgentArtifactError(
            f"agent runtime signature missing for {strategy} seed {seed}"
        )
    _, stored_runtime_hash = _validated_training_runtime_signature(
        stored_signature, label=f"agent record {strategy} seed {seed}"
    )
    if _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256:
        _validate_production_determinism_signature(stored_signature)
    if record.get("training_runtime_signature_sha256") != stored_runtime_hash:
        raise CorruptAgentArtifactError(
            f"agent runtime signature hash mismatch for {strategy} seed {seed}"
        )
    current_runtime_signature, current_runtime_hash = _current_training_runtime()
    if stored_runtime_hash != current_runtime_hash:
        raise AgentContractDriftError(
            f"training runtime drift for {strategy} seed {seed}"
        )
    expected_contract = _agent_contract_hash(
        config, strategy, seed, implementation_hash, stored_signature
    )
    if record.get("agent_contract_hash") != expected_contract:
        raise AgentContractDriftError(
            f"agent contract drift for {strategy} seed {seed}"
        )
    production_agent = _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256
    stored_gpu_attestation = record.get("gpu_load_attestation")
    stored_gpu_binding = record.get("gpu_load_attestation_binding")
    if production_agent:
        try:
            normalized_gpu_attestation = _validate_gpu_load_attestation(
                stored_gpu_attestation,
                config_sha256=_canonical_sha256(config),
                implementation_sha256=implementation_hash,
                expected_purpose="TRAINING_LAUNCH",
                current_runtime=current_runtime_signature,
                require_fresh=False,
                require_live_parent=False,
            )
            normalized_gpu_binding = _validated_gpu_attestation_binding(
                stored_gpu_binding
            )
        except GpuLoadAttestationError as error:
            raise CorruptAgentArtifactError(
                f"agent GPU attestation invalid for {strategy} seed {seed}: {error}"
            ) from error
        if normalized_gpu_binding != _gpu_attestation_binding(
            normalized_gpu_attestation
        ):
            raise CorruptAgentArtifactError(
                f"agent GPU attestation binding mismatch for {strategy} seed {seed}"
            )
    elif stored_gpu_attestation is not None or stored_gpu_binding is not None:
        raise CorruptAgentArtifactError(
            f"non-production agent carried a production GPU attestation for {strategy} seed {seed}"
        )
    if (
        record.get("schema_version") != "t9.1.3-agent-record-v1"
        or record.get("task_id") != TASK_ID
        or record.get("evidence_grade") != EVIDENCE_GRADE
        or record.get("strategy") != strategy
        or int(record.get("root_seed", -1)) != int(seed)
        or record.get("config_sha256") != _canonical_sha256(config)
        or record.get("implementation_sha256") != implementation_hash
        or record.get("parent_analysis_sha256") != T9_1_1_ANALYSIS_SHA256
        or _canonical_sha256(record.get("strategy_config"))
        != _canonical_sha256(asdict(directional))
    ):
        raise CorruptAgentArtifactError(
            f"agent record provenance mismatch for {strategy} seed {seed}"
        )
    if _file_sha256(weight_path) != record.get("weight_file_sha256"):
        raise CorruptAgentArtifactError(
            f"weight file hash mismatch for {strategy} seed {seed}"
        )
    if _file_sha256(terminal_path) != record.get("terminal_file_sha256"):
        raise CorruptAgentArtifactError(
            f"terminal restart hash mismatch for {strategy} seed {seed}"
        )
    prior_failure = record.get("prior_failure_record")
    if (
        record.get("weight_path") != _portable_artifact_path(weight_path)
        or int(record.get("weight_bytes", -1)) != weight_path.stat().st_size
        or record.get("terminal_path") != _portable_artifact_path(terminal_path)
        or int(record.get("terminal_bytes", -1)) != terminal_path.stat().st_size
        or (
            prior_failure is not None
            and (
                not isinstance(prior_failure, str)
                or not _resolve_repo_path(prior_failure).is_file()
            )
        )
    ):
        raise CorruptAgentArtifactError(
            f"agent path/byte provenance mismatch for {strategy} seed {seed}"
        )
    attempt_id = record.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise CorruptAgentArtifactError(
            f"agent attempt identity missing for {strategy} seed {seed}"
        )
    model, tensor_hash = _load_npz_model(
        weight_path,
        strategy,
        directional,
        seed,
        expected_attempt_id=attempt_id,
    )
    if tensor_hash != record.get("canonical_tensor_sha256"):
        raise CorruptAgentArtifactError(
            f"canonical tensor hash mismatch for {strategy} seed {seed}"
        )
    torch_hash = state_dict_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    if torch_hash != record["training_record"]["checkpoint_sha256"]:
        raise CorruptAgentArtifactError(
            f"reload witness mismatch for {strategy} seed {seed}"
        )
    training_record = record["training_record"]
    expected_epochs = int(config["training"]["epochs"])
    expected_training_record_fields = {
        "strategy",
        "training_seed",
        "parameter_count",
        "architecture",
        "epochs_executed",
        "training_trajectories",
        "baseline_warmup_trajectories",
        "baseline_initialization",
        "baseline_warmup_seed",
        "training_trajectory_seeds",
        "validation_seeds_used_for_checkpoint_selection_only",
        "best_validation_epoch",
        "best_validation_score",
        "initial_validation_score",
        "validation_history",
        "training_curve",
        "checkpoint_sha256",
        "initial_state_sha256",
        "terminal_state_sha256",
        "training_lineage_schema_version",
        "training_lineage_genesis_sha256",
        "training_lineage_terminal_sha256",
        "training_lineage_complete",
        "agent_local_rng_contract",
        "continuous_adam_optimizer",
        "terminal_optimizer_state_retained",
        "terminal_rng_state_retained",
        "all_validation_candidate_states_retained",
        "validation_raw_per_seed_and_trajectory_retained",
        "wall_time_seconds",
    }
    expected_architecture = (
        PAPER_MF_ARCHITECTURE if strategy == "mf" else PAPER_RNN_ARCHITECTURE
    )
    expected_parameter_count = 70_159 if strategy == "mf" else 72_853
    expected_training_seeds = [
        _training_trajectory_seed(seed, epoch)
        for epoch in range(1, expected_epochs + 1)
    ]
    if (
        set(training_record) != expected_training_record_fields
        or training_record.get("strategy") != strategy
        or int(training_record.get("training_seed", -1)) != int(seed)
        or training_record.get("architecture") != expected_architecture
        or int(training_record.get("parameter_count", -1))
        != expected_parameter_count
        or int(training_record.get("epochs_executed", -1)) != expected_epochs
        or [int(row.get("epoch", -1)) for row in training_record.get("training_curve", ())]
        != list(range(1, expected_epochs + 1))
        or training_record.get("training_trajectory_seeds")
        != expected_training_seeds
        or training_record.get("validation_seeds_used_for_checkpoint_selection_only")
        != list(directional.validation_seeds)
        or training_record.get("continuous_adam_optimizer") is not True
        or training_record.get("terminal_optimizer_state_retained") is not True
        or training_record.get("terminal_rng_state_retained") is not True
        or training_record.get("all_validation_candidate_states_retained") is not True
        or training_record.get("validation_raw_per_seed_and_trajectory_retained")
        is not True
        or not all(
            set(row) == set(TRAINING_CURVE_ROW_KEYS)
            and _all_numeric_finite(
                {name: row[name] for name in TRAINING_METRIC_KEYS}
            )
            and float(row["gradient_norm_before_clip"]) > 0.0
            for row in training_record.get("training_curve", ())
        )
        or not all(
            _is_lower_sha256(training_record.get(name))
            for name in (
                "checkpoint_sha256",
                "initial_state_sha256",
                "terminal_state_sha256",
                "training_lineage_genesis_sha256",
                "training_lineage_terminal_sha256",
            )
        )
        or not np.isfinite(float(training_record.get("wall_time_seconds", np.nan)))
        or float(training_record.get("wall_time_seconds", -1.0)) < 0.0
    ):
        raise CorruptAgentArtifactError(
            f"training/validation ledger mismatch for {strategy} seed {seed}"
        )
    selected_history = [
        point
        for point in training_record["validation_history"]
        if int(point["epoch"]) == int(training_record["best_validation_epoch"])
    ]
    if (
        len(selected_history) != 1
        or selected_history[0].get("candidate_state_sha256") != tensor_hash
        or not np.isclose(
            float(selected_history[0]["selection_score"]),
            float(training_record["best_validation_score"]),
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise CorruptAgentArtifactError(
            f"validation-selected history mismatch for {strategy} seed {seed}"
        )
    try:
        terminal = th.load(terminal_path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError, ValueError, TypeError) as error:
        raise CorruptAgentArtifactError(
            f"terminal restart cannot be decoded for {strategy} seed {seed}"
        ) from error
    if not isinstance(terminal, dict):
        raise CorruptAgentArtifactError(
            f"terminal restart is not a mapping for {strategy} seed {seed}"
        )
    if set(terminal) != {
        "schema_version",
        "strategy",
        "root_seed",
        "terminal_epoch",
        "terminal_model_state",
        "terminal_model_sha256",
        "validation_selected_state",
        "validation_selected_sha256",
        "validation_candidate_states",
        "optimizer_state",
        "score_baseline",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "numpy_random_state",
        "training_lineage_schema_version",
        "training_lineage_genesis_sha256",
        "training_lineage_terminal_sha256",
        "agent_local_rng_contract",
        "agent_contract_hash",
        "training_runtime_signature",
        "training_runtime_signature_sha256",
        "config_sha256",
        "implementation_sha256",
        "parent_analysis_sha256",
        "gpu_load_attestation",
        "gpu_load_attestation_binding",
        "attempt_id",
    }:
        raise CorruptAgentArtifactError(
            f"terminal restart schema drift for {strategy} seed {seed}"
        )
    if terminal.get("agent_contract_hash") != expected_contract:
        raise AgentContractDriftError(
            f"terminal contract drift for {strategy} seed {seed}"
        )
    rng_contract = _validated_agent_local_rng_contract(
        training_record.get("agent_local_rng_contract"),
        strategy=strategy,
        seed=seed,
    )
    if terminal.get("agent_local_rng_contract") != rng_contract:
        raise CorruptAgentArtifactError(
            f"terminal/record agent-local RNG mismatch for {strategy} seed {seed}"
        )
    terminal_signature = terminal.get("training_runtime_signature")
    if not isinstance(terminal_signature, Mapping):
        raise CorruptAgentArtifactError(
            f"terminal runtime signature missing for {strategy} seed {seed}"
        )
    _, terminal_runtime_hash = _validated_training_runtime_signature(
        terminal_signature, label=f"terminal restart {strategy} seed {seed}"
    )
    if (
        terminal_runtime_hash != stored_runtime_hash
        or dict(terminal_signature) != dict(stored_signature)
        or terminal.get("training_runtime_signature_sha256")
        != stored_runtime_hash
        or terminal.get("gpu_load_attestation") != stored_gpu_attestation
        or terminal.get("gpu_load_attestation_binding") != stored_gpu_binding
    ):
        raise CorruptAgentArtifactError(
            f"terminal/agent runtime signature mismatch for {strategy} seed {seed}"
        )
    try:
        selected_terminal_hash = state_dict_sha256(
            terminal["validation_selected_state"]
        )
        actual_terminal_hash = state_dict_sha256(terminal["terminal_model_state"])
    except (KeyError, TypeError, AttributeError) as error:
        raise CorruptAgentArtifactError(
            f"terminal tensor payload invalid for {strategy} seed {seed}"
        ) from error
    if (
        terminal.get("validation_selected_sha256") != tensor_hash
        or selected_terminal_hash != tensor_hash
    ):
        raise CorruptAgentArtifactError(
            f"terminal/selected checkpoint mismatch for {strategy} seed {seed}"
        )
    if (
        terminal.get("terminal_model_sha256")
        != record["training_record"]["terminal_state_sha256"]
        or actual_terminal_hash != terminal.get("terminal_model_sha256")
    ):
        raise CorruptAgentArtifactError(
            f"terminal model hash mismatch for {strategy} seed {seed}"
        )
    reference_state = model.state_dict()
    _assert_finite_state_dict(
        terminal["validation_selected_state"],
        reference_state,
        label=f"validation-selected state for {strategy} seed {seed}",
    )
    _assert_finite_state_dict(
        terminal["terminal_model_state"],
        reference_state,
        label=f"terminal state for {strategy} seed {seed}",
    )
    validation_physics_replay = _validate_validation_evidence(
        training_record=training_record,
        terminal=terminal,
        model=model,
        directional=directional,
        strategy=strategy,
        seed=seed,
        replay_physics=replay_validation_candidates,
    )
    optimizer_state = terminal.get("optimizer_state")
    cpu_rng = terminal.get("torch_cpu_rng_state")
    numpy_rng = terminal.get("numpy_random_state")
    cuda_rng = terminal.get("torch_cuda_rng_states")
    optimizer_steps: list[float] = []
    if isinstance(optimizer_state, dict):
        for value in optimizer_state.get("state", {}).values():
            if isinstance(value, dict) and "step" in value:
                step = value["step"]
                optimizer_steps.append(
                    float(step.detach().cpu()) if hasattr(step, "detach") else float(step)
                )
    _validate_optimizer_restart(
        optimizer_state,
        model,
        expected_epochs=expected_epochs,
        expected_learning_rate=float(directional.learning_rate),
        strategy=strategy,
        seed=seed,
    )
    stored_device_count = int(stored_signature["cuda_device_count"])
    restart_valid = (
        terminal.get("schema_version") == "t9.1.3-terminal-restart-v1"
        and terminal.get("attempt_id") == attempt_id
        and terminal.get("strategy") == strategy
        and int(terminal.get("root_seed", -1)) == int(seed)
        and int(terminal.get("terminal_epoch", -1)) == expected_epochs
        and terminal.get("config_sha256") == _canonical_sha256(config)
        and terminal.get("implementation_sha256") == implementation_hash
        and isinstance(optimizer_state, dict)
        and bool(optimizer_state.get("state"))
        and bool(optimizer_state.get("param_groups"))
        and bool(optimizer_steps)
        and all(step == float(expected_epochs) for step in optimizer_steps)
        and isinstance(cpu_rng, th.Tensor)
        and cpu_rng.dtype == th.uint8
        and cpu_rng.ndim == 1
        and int(cpu_rng.numel()) > 0
        and isinstance(numpy_rng, tuple)
        and len(numpy_rng) == 5
        and numpy_rng[0] == "MT19937"
        and isinstance(numpy_rng[1], np.ndarray)
        and numpy_rng[1].dtype == np.uint32
        and numpy_rng[1].shape == (624,)
        and isinstance(numpy_rng[2], (int, np.integer))
        and 0 <= int(numpy_rng[2]) <= 624
        and int(numpy_rng[3]) in {0, 1}
        and np.isfinite(float(numpy_rng[4]))
        and isinstance(cuda_rng, list)
        and all(
            isinstance(value, th.Tensor)
            and value.dtype == th.uint8
            and value.ndim == 1
            and value.numel() > 0
            for value in cuda_rng
        )
        and (
            str(config["training"]["device"]) != "cuda"
            or len(cuda_rng) == stored_device_count > 0
        )
        and np.isfinite(float(terminal.get("score_baseline", np.nan)))
        and np.isclose(
            float(terminal["score_baseline"]),
            float(training_record["training_curve"][-1]["baseline"]),
            rtol=0.0,
            atol=0.0,
        )
    )
    if not restart_valid:
        raise CorruptAgentArtifactError(
            f"terminal optimizer/RNG state missing for {strategy} seed {seed}"
        )
    training_lineage_validation = _validate_training_lineage(
        training_record=training_record,
        terminal=terminal,
        model=model,
        directional=directional,
        strategy=strategy,
        seed=seed,
    )
    witness_changed = (
        not record.get("reload_witness")
        or not record.get("terminal_reload_witness")
        or not record.get("validation_candidate_reload_witness")
        or not record.get("training_lineage_reload_witness")
    )
    record["reload_witness"] = True
    record["terminal_reload_witness"] = True
    record["validation_candidate_reload_witness"] = True
    record["training_lineage_reload_witness"] = True
    if witness_changed:
        _atomic_json(record, record_path)
    if validation_physics_replay is not None:
        record["live_validation_candidate_physics_replay"] = (
            validation_physics_replay
        )
    record["live_training_lineage_validation"] = training_lineage_validation
    return model, record


def _publish_agent_bundle(
    *,
    output_dir: Path,
    strategy: Literal["mf", "nmf"],
    seed: int,
    directional: DirectionalRankingConfig,
    config: Mapping[str, Any],
    implementation_hash: str,
    model: Any,
    training_record: Mapping[str, Any],
    terminal_payload: Mapping[str, Any],
    prior_failure_path: Path | None,
    attempt_id: str,
    gpu_load_attestation: Mapping[str, Any] | None,
    gpu_load_attestation_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Publish NPZ/PT/JSON in commit-marker order and prove a clean reload."""

    arrays = _state_numpy(model)
    tensor_hash = _numpy_state_sha256(arrays)
    if tensor_hash != training_record["checkpoint_sha256"]:
        raise RuntimeError("NumPy and torch canonical checkpoint hashes disagree")
    weight_path, terminal_path, record_path = _agent_paths(
        output_dir, strategy, seed
    )
    runtime_signature, runtime_signature_hash = _current_training_runtime()
    contract_hash = _agent_contract_hash(
        config, strategy, seed, implementation_hash, runtime_signature
    )
    start_path = output_dir / "_training_attempts" / attempt_id / "start.json"
    try:
        start = _load_json(start_path)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise AgentArtifactError("training attempt start marker is unavailable") from error
    if (
        start.get("attempt_id") != attempt_id
        or start.get("strategy") != strategy
        or int(start.get("root_seed", -1)) != int(seed)
        or start.get("agent_contract_hash") != contract_hash
        or start.get("training_runtime_signature_sha256")
        != runtime_signature_hash
        or start.get("training_runtime_signature") != runtime_signature
        or start.get("runtime_signature_state") != "CAPTURED_BEFORE_TRAINING"
        or start.get("gpu_load_attestation_binding")
        != (
            dict(gpu_load_attestation_binding)
            if gpu_load_attestation_binding is not None
            else None
        )
    ):
        raise AgentContractDriftError("training attempt/start runtime contract drifted")
    terminal = dict(terminal_payload)
    terminal["agent_contract_hash"] = contract_hash
    terminal["training_runtime_signature"] = runtime_signature
    terminal["training_runtime_signature_sha256"] = runtime_signature_hash
    terminal["config_sha256"] = _canonical_sha256(config)
    terminal["implementation_sha256"] = implementation_hash
    terminal["parent_analysis_sha256"] = T9_1_1_ANALYSIS_SHA256
    terminal["attempt_id"] = attempt_id
    terminal["gpu_load_attestation"] = (
        dict(gpu_load_attestation) if gpu_load_attestation is not None else None
    )
    terminal["gpu_load_attestation_binding"] = (
        dict(gpu_load_attestation_binding)
        if gpu_load_attestation_binding is not None
        else None
    )

    # The record JSON is the bundle's commit marker and is therefore written
    # last.  Any interruption before it appears is recognized as a partial shard.
    _atomic_npz(
        {
            **arrays,
            NPZ_ATTEMPT_ID_FIELD: np.asarray(attempt_id),
        },
        weight_path,
    )
    _atomic_torch(terminal, terminal_path)
    record = {
        "schema_version": "t9.1.3-agent-record-v1",
        "task_id": TASK_ID,
        "evidence_grade": EVIDENCE_GRADE,
        "strategy": strategy,
        "root_seed": int(seed),
        "config_sha256": _canonical_sha256(config),
        "agent_contract_hash": contract_hash,
        "training_runtime_signature": runtime_signature,
        "training_runtime_signature_sha256": runtime_signature_hash,
        "implementation_sha256": implementation_hash,
        "parent_analysis_sha256": T9_1_1_ANALYSIS_SHA256,
        "gpu_load_attestation": (
            dict(gpu_load_attestation) if gpu_load_attestation is not None else None
        ),
        "gpu_load_attestation_binding": (
            dict(gpu_load_attestation_binding)
            if gpu_load_attestation_binding is not None
            else None
        ),
        "attempt_id": attempt_id,
        "weight_path": _portable_artifact_path(weight_path),
        "weight_file_sha256": _file_sha256(weight_path),
        "canonical_tensor_sha256": tensor_hash,
        "weight_bytes": weight_path.stat().st_size,
        "terminal_path": _portable_artifact_path(terminal_path),
        "terminal_file_sha256": _file_sha256(terminal_path),
        "terminal_bytes": terminal_path.stat().st_size,
        "prior_failure_record": (
            _portable_artifact_path(prior_failure_path)
            if prior_failure_path is not None and prior_failure_path.exists()
            else None
        ),
        "strategy_config": asdict(directional),
        "training_record": dict(training_record),
        "reload_witness": False,
        "terminal_reload_witness": False,
        "validation_candidate_reload_witness": False,
        "training_lineage_reload_witness": False,
    }
    _atomic_json(record, record_path)
    loaded = _load_agent_record(
        output_dir=output_dir,
        strategy=strategy,
        seed=seed,
        directional=directional,
        config=config,
        implementation_hash=implementation_hash,
    )
    if loaded is None:
        raise RuntimeError("newly written agent could not be reloaded")
    return loaded[1]


def _failure_paths(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.rglob("failure.json")
        if path.is_file()
    ) + sorted(
        path
        for path in output_dir.rglob("*.failure.json")
        if path.is_file()
    )


def _latest_failure_for_agent(
    output_dir: Path, strategy: str, seed: int
) -> Path | None:
    matches: list[Path] = []
    for path in _failure_paths(output_dir):
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            continue
        if (
            payload.get("strategy") == strategy
            and int(payload.get("root_seed", -1)) == int(seed)
        ):
            matches.append(path)
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None


def _train_or_resume_agent(
    *,
    config: Mapping[str, Any],
    destination: Path,
    strategy: Literal["mf", "nmf"],
    seed: int,
    directional: DirectionalRankingConfig,
    implementation_hash: str,
    gpu_load_attestation: Mapping[str, Any] | None,
    gpu_load_attestation_binding: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Return a verified record and whether this invocation trained it."""

    _recover_interrupted_training_attempts(
        output_dir=destination,
        strategy=strategy,
        seed=seed,
        config=config,
        directional=directional,
        implementation_hash=implementation_hash,
    )
    try:
        loaded = _load_agent_record(
            output_dir=destination,
            strategy=strategy,
            seed=seed,
            directional=directional,
            config=config,
            implementation_hash=implementation_hash,
        )
    except AgentContractDriftError:
        raise
    except AgentArtifactError as error:
        archive_path = _archive_incomplete_agent_bundle(
            output_dir=destination,
            strategy=strategy,
            seed=seed,
            config=config,
            implementation_hash=implementation_hash,
            error=error,
            traceback_text=traceback.format_exc(),
        )
        print(
            _canonical_json(
                {
                    "event": "t9_1_3_corrupt_bundle_archived",
                    "strategy": strategy,
                    "seed": seed,
                    "archive_failure_path": archive_path.as_posix(),
                }
            ),
            flush=True,
        )
        loaded = None
    if loaded is not None:
        return loaded[1], False

    prior_failure_path = _latest_failure_for_agent(destination, strategy, seed)
    attempt_directory = _start_training_attempt(
        output_dir=destination,
        strategy=strategy,
        seed=seed,
        config=config,
        implementation_hash=implementation_hash,
        gpu_load_attestation_binding=gpu_load_attestation_binding,
    )
    attempt_id = _attempt_id(attempt_directory)
    try:
        model, training_record, terminal_payload = _train_paper_agent(
            strategy, seed, directional
        )
    except Exception as error:
        message = str(error)
        lowered = message.lower()
        failure_state = (
            "OOM"
            if "out of memory" in lowered
            else "TIMEOUT"
            if "timeout" in lowered or "timed out" in lowered
            else "NUMERIC_FAILURE"
            if "non-finite" in lowered or "invalid gradient" in lowered
            else "NULL"
        )
        failure_path = attempt_directory / "failure.json"
        failure = _failure_record(
            config=config,
            implementation_hash=implementation_hash,
            strategy=strategy,
            seed=seed,
            failure_state=failure_state,
            failure_subtype=(
                type(error).__name__
                if failure_state != "NULL"
                else f"RUNTIME_FAILURE:{type(error).__name__}"
            ),
            exception_type=type(error).__name__,
            exception_message=message,
            traceback_text=traceback.format_exc(),
            safe_retry_authorized=True,
        )
        failure["training_attempt_start_path"] = (
            _portable_artifact_path(attempt_directory / "start.json")
        )
        failure["attempt_id"] = attempt_id
        _atomic_json(failure, failure_path)
        print(
            _canonical_json(
                {
                    "event": "t9_1_3_agent_failure_retained",
                    "strategy": strategy,
                    "seed": seed,
                    "failure_state": failure_state,
                    "failure_path": failure_path.as_posix(),
                }
            ),
            flush=True,
        )
        raise

    record: dict[str, Any] | None = None
    for publication_attempt in (1, 2):
        try:
            record = _publish_agent_bundle(
                output_dir=destination,
                strategy=strategy,
                seed=seed,
                directional=directional,
                config=config,
                implementation_hash=implementation_hash,
                model=model,
                training_record=training_record,
                terminal_payload=terminal_payload,
                prior_failure_path=prior_failure_path,
                attempt_id=attempt_id,
                gpu_load_attestation=gpu_load_attestation,
                gpu_load_attestation_binding=gpu_load_attestation_binding,
            )
            break
        except Exception as error:
            archive_path = _archive_incomplete_agent_bundle(
                output_dir=destination,
                strategy=strategy,
                seed=seed,
                config=config,
                implementation_hash=implementation_hash,
                error=error,
                traceback_text=traceback.format_exc(),
            )
            print(
                _canonical_json(
                    {
                        "event": "t9_1_3_publication_retry",
                        "strategy": strategy,
                        "seed": seed,
                        "publication_attempt": publication_attempt,
                        "archive_failure_path": archive_path.as_posix(),
                    }
                ),
                flush=True,
            )
            if publication_attempt == 2:
                failure = _failure_record(
                    config=config,
                    implementation_hash=implementation_hash,
                    strategy=strategy,
                    seed=seed,
                    failure_state="NULL",
                    failure_subtype="PUBLICATION_FAILED_AFTER_RETRY",
                    exception_type=type(error).__name__,
                    exception_message=str(error),
                    traceback_text=traceback.format_exc(),
                    safe_retry_authorized=True,
                )
                failure["training_attempt_start_path"] = (
                    _portable_artifact_path(attempt_directory / "start.json")
                )
                failure["attempt_id"] = attempt_id
                _atomic_json(failure, attempt_directory / "failure.json")
                raise
    if record is None:  # pragma: no cover - loop either returns or raises.
        raise RuntimeError("agent publication exhausted without a record")
    _atomic_json(
        {
            "schema_version": "t9.1.3-training-attempt-outcome-v1",
            "task_id": TASK_ID,
            "strategy": strategy,
            "root_seed": int(seed),
            "attempt_id": attempt_id,
            "state": "SUCCESS",
            "agent_contract_hash": record["agent_contract_hash"],
            "training_runtime_signature_sha256": record[
                "training_runtime_signature_sha256"
            ],
            "gpu_load_attestation_binding": record[
                "gpu_load_attestation_binding"
            ],
            "canonical_tensor_sha256": record["canonical_tensor_sha256"],
            "weight_file_sha256": record["weight_file_sha256"],
            "terminal_file_sha256": record["terminal_file_sha256"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        attempt_directory / "outcome.json",
    )
    print(
        _canonical_json(
            {
                "event": "t9_1_3_agent_published",
                "strategy": strategy,
                "seed": seed,
                "best_epoch": training_record["best_validation_epoch"],
                "weight_sha256": record["weight_file_sha256"],
            }
        ),
        flush=True,
    )
    return record, True


def _serial_release_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must carry UTC offset zero")
    return parsed.astimezone(timezone.utc)


def _serial_release_expected_path(
    gate_path: str | Path, gpu_attestation: Mapping[str, Any]
) -> tuple[Path, Path]:
    run_identity = gpu_attestation.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise ValueError("serial-release attestation run identity is missing")
    run_dir_value = run_identity.get("run_dir")
    if not isinstance(run_dir_value, str) or not Path(run_dir_value).is_absolute():
        raise ValueError("serial-release attested run_dir must be absolute")
    candidate = Path(gate_path)
    if not candidate.is_absolute():
        raise ValueError("serial-release gate path must be absolute")
    run_dir = Path(run_dir_value).resolve(strict=False)
    expected = (run_dir / SERIAL_RELEASE_FILENAME).resolve(strict=False)
    observed = candidate.resolve(strict=False)
    if os.path.normcase(str(observed)) != os.path.normcase(str(expected)):
        raise ValueError(
            "serial-release gate path is not the unique file bound to attested run_dir"
        )
    return observed, run_dir


def _validate_nmf_serial_release_payload(
    payload: Any,
    *,
    gate_path: Path,
    run_dir: Path,
    release_nonce: str,
    deadline_utc: str,
    wait_started_at: datetime,
    config_sha256: str,
    implementation_sha256: str,
    gpu_attestation: Mapping[str, Any],
    waiter_pid: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != SERIAL_RELEASE_KEYS:
        raise ValueError("NMF serial-release payload schema drifted")
    release = dict(payload)
    run_identity = gpu_attestation["run_identity"]
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    released_at = _serial_release_utc(release.get("released_at_utc"), "released_at_utc")
    parsed_deadline = _serial_release_utc(deadline_utc, "serial-release deadline")
    try:
        canonical_nonce = str(uuid.UUID(release_nonce))
    except (ValueError, AttributeError) as error:
        raise ValueError("serial-release nonce must be a canonical UUID") from error
    if canonical_nonce != release_nonce.lower():
        raise ValueError("serial-release nonce must use canonical UUID spelling")
    if (
        release.get("schema_version") != SERIAL_RELEASE_SCHEMA_VERSION
        or release.get("task_id") != TASK_ID
        or release.get("family") != "nmf"
        or release.get("prerequisite_family") != "mf"
        or isinstance(release.get("prerequisite_exit_code"), bool)
        or release.get("prerequisite_exit_code") != 0
        or release.get("transaction_id") != run_identity.get("transaction_id")
        or release.get("run_dir") != str(run_dir)
        or release.get("attestation_nonce")
        != gpu_attestation.get("attestation_nonce")
        or release.get("attestation_sha256")
        != gpu_attestation.get("attestation_sha256")
        or release.get("release_nonce") != release_nonce
        or isinstance(release.get("waiter_pid"), bool)
        or release.get("waiter_pid") != waiter_pid
        or release.get("config_sha256") != config_sha256
        or release.get("implementation_sha256") != implementation_sha256
        or release.get("deadline_utc") != deadline_utc
        or isinstance(release.get("mf_pid"), bool)
        or not isinstance(release.get("mf_pid"), int)
        or release["mf_pid"] <= 0
        or isinstance(release.get("mf_process_created_unix_ns"), bool)
        or not isinstance(release.get("mf_process_created_unix_ns"), int)
        or release["mf_process_created_unix_ns"] <= 0
        or released_at < wait_started_at
        or released_at > observed_now + timedelta(seconds=5)
        or released_at > parsed_deadline
        or os.path.normcase(str(gate_path.parent)) != os.path.normcase(str(run_dir))
    ):
        raise ValueError("NMF serial-release payload identity or timeline mismatch")
    return release


def _wait_for_nmf_serial_release(
    *,
    gate_path: str | Path,
    release_nonce: str,
    deadline_utc: str,
    config_sha256: str,
    implementation_sha256: str,
    gpu_attestation: Mapping[str, Any],
    poll_interval_seconds: float = 0.05,
    emit_ready: bool = True,
) -> dict[str, Any]:
    """Wait before touching the output namespace for a supervisor MF-success seal."""

    if not isinstance(release_nonce, str):
        raise ValueError("serial-release nonce must be a string")
    try:
        if str(uuid.UUID(release_nonce)) != release_nonce.lower():
            raise ValueError
    except (ValueError, AttributeError) as error:
        raise ValueError("serial-release nonce must be a canonical UUID") from error
    if not isinstance(deadline_utc, str):
        raise ValueError("serial-release deadline must be a string")
    if not isinstance(poll_interval_seconds, (int, float)) or not (
        0.001 <= float(poll_interval_seconds) <= 1.0
    ):
        raise ValueError("serial-release poll interval is outside the safe range")
    if not isinstance(emit_ready, bool):
        raise ValueError("serial-release ready emission flag must be boolean")
    path, run_dir = _serial_release_expected_path(gate_path, gpu_attestation)
    deadline = _serial_release_utc(deadline_utc, "serial-release deadline")
    wait_started_at = datetime.now(timezone.utc)
    if deadline <= wait_started_at:
        raise TimeoutError("NMF serial-release deadline expired before waiting")
    # lexists also rejects a broken symlink.  A release created before this
    # process reaches the post-attestation barrier can never authorize work.
    if os.path.lexists(path):
        raise ValueError("NMF serial-release gate pre-existed the validated wait")
    wait_started_monotonic = time.monotonic()
    monotonic_deadline = wait_started_monotonic + (
        deadline - wait_started_at
    ).total_seconds()
    ready = {
        "schema_version": SERIAL_RELEASE_READY_SCHEMA_VERSION,
        "event": "t9_1_3_nmf_serial_release_wait_ready",
        "task_id": TASK_ID,
        "family": "nmf",
        "waiter_pid": os.getpid(),
        "release_path": str(path),
        "release_nonce_sha256": hashlib.sha256(
            release_nonce.encode("utf-8")
        ).hexdigest(),
        "transaction_id": gpu_attestation["run_identity"]["transaction_id"],
        "attestation_nonce": gpu_attestation["attestation_nonce"],
        "attestation_sha256": gpu_attestation["attestation_sha256"],
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "deadline_utc": deadline_utc,
    }
    if emit_ready:
        print(_canonical_json(ready), flush=True)
    while not os.path.lexists(path):
        if time.monotonic() >= monotonic_deadline or datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("timed out waiting for the NMF serial-release gate")
        time.sleep(float(poll_interval_seconds))
    if time.monotonic() >= monotonic_deadline or datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("NMF serial-release gate arrived after the transaction deadline")
    if path.is_symlink() or not path.is_file():
        raise ValueError("NMF serial-release gate is not a regular file")
    try:
        raw_release = path.read_bytes()
        if len(raw_release) > 65536:
            raise ValueError("NMF serial-release gate exceeds the size bound")
        payload = json.loads(raw_release.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("NMF serial-release gate is unreadable") from error
    release = _validate_nmf_serial_release_payload(
        payload,
        gate_path=path,
        run_dir=run_dir,
        release_nonce=release_nonce,
        deadline_utc=deadline_utc,
        wait_started_at=wait_started_at,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        gpu_attestation=gpu_attestation,
        waiter_pid=os.getpid(),
    )
    if time.monotonic() >= monotonic_deadline or datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("NMF serial-release validation crossed the transaction deadline")
    witness = {
        "schema_version": "t9.1.3-nmf-serial-release-witness-v1",
        "event": "t9_1_3_nmf_serial_release_consumed",
        "release_path": str(path),
        "release_sha256": hashlib.sha256(raw_release).hexdigest(),
        "transaction_id": release["transaction_id"],
        "attestation_sha256": release["attestation_sha256"],
        "release_nonce_sha256": hashlib.sha256(
            release_nonce.encode("utf-8")
        ).hexdigest(),
        "mf_pid": release["mf_pid"],
        "mf_process_created_unix_ns": release["mf_process_created_unix_ns"],
        "waiter_pid": os.getpid(),
        "released_at_utc": release["released_at_utc"],
        "waited_seconds": time.monotonic() - wait_started_monotonic,
    }
    if emit_ready:
        print(_canonical_json(witness), flush=True)
    return witness


def train_population(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    family: Literal["mf", "nmf", "all"] = "all",
    production: bool = True,
    gpu_attestation: Mapping[str, Any] | str | Path | None = None,
    serial_release_gate: str | Path | None = None,
    serial_release_nonce: str | None = None,
    serial_release_deadline_utc: str | None = None,
) -> dict[str, Any]:
    """Train every requested agent and atomically publish one checkpoint per root."""

    _reject_canonical_production_config_downgrade(config, production=production)
    if production:
        _configure_production_determinism()
    _require_torch()
    _validate_config(config, production=production)
    canonical_production = (
        production and _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256
    )
    release_arguments = (
        serial_release_gate,
        serial_release_nonce,
        serial_release_deadline_utc,
    )
    if canonical_production and family == "all":
        raise ValueError(
            "canonical production family=all is forbidden; the supervisor must "
            "own distinct MF and release-gated NMF children"
        )
    if (
        canonical_production
        and family == "nmf"
        and not all(value is not None for value in release_arguments)
    ):
        raise ValueError(
            "canonical production NMF requires the supervisor MF-success "
            "serial-release gate"
        )
    runtime_signature, _ = _validate_runtime_for_config(
        config, production=production
    )
    _verify_parent_protocol(config)
    implementation_hash = implementation_sha256()
    normalized_gpu_attestation: dict[str, Any] | None = None
    gpu_attestation_binding: dict[str, Any] | None = None
    if production:
        normalized_gpu_attestation = _validate_gpu_load_attestation(
            gpu_attestation,
            config_sha256=_canonical_sha256(config),
            implementation_sha256=implementation_hash,
            expected_purpose="TRAINING_LAUNCH",
            current_runtime=runtime_signature,
            require_fresh=True,
            require_live_parent=True,
        )
        gpu_attestation_binding = _gpu_attestation_binding(
            normalized_gpu_attestation
        )
    serial_release_witness: dict[str, Any] | None = None
    if any(value is not None for value in release_arguments):
        if not all(value is not None for value in release_arguments):
            raise ValueError("the NMF serial-release gate requires all three fields")
        if not production or family != "nmf" or normalized_gpu_attestation is None:
            raise ValueError(
                "the NMF serial-release gate is restricted to a supervised "
                "production NMF-only launch"
            )
        serial_release_witness = _wait_for_nmf_serial_release(
            gate_path=serial_release_gate,
            release_nonce=serial_release_nonce,
            deadline_utc=serial_release_deadline_utc,
            config_sha256=_canonical_sha256(config),
            implementation_sha256=implementation_hash,
            gpu_attestation=normalized_gpu_attestation,
        )
    destination = Path(output_dir)
    if production:
        _require_canonical_production_path(
            destination, "docs/t9_1_3_puviani_agents"
        )
    _assert_finalize_not_active(destination)
    strategies: tuple[Literal["mf", "nmf"], ...] = (
        ("mf", "nmf") if family == "all" else (family,)
    )
    started = time.perf_counter()
    trained = 0
    resumed = 0
    agents: list[dict[str, Any]] = []
    for strategy in strategies:
        directional = _strategy_config(config, strategy)
        for seed in directional.training_seeds:
            _assert_finalize_not_active(destination)
            with _agent_writer_lock(
                output_dir=destination,
                strategy=strategy,
                seed=seed,
                config=config,
                implementation_hash=implementation_hash,
            ):
                _assert_finalize_not_active(destination)
                _quarantine_agent_atomic_temporaries(
                    output_dir=destination,
                    strategy=strategy,
                    seed=int(seed),
                    config=config,
                    implementation_hash=implementation_hash,
                )
                _recover_incomplete_failed_attempts(
                    destination,
                    config,
                    implementation_hash,
                    strategy_filter=strategy,
                    seed_filter=int(seed),
                )
                _recover_incomplete_archives(
                    destination,
                    config,
                    implementation_hash,
                    strategy_filter=strategy,
                    seed_filter=int(seed),
                )
                record, trained_now = _train_or_resume_agent(
                    config=config,
                    destination=destination,
                    strategy=strategy,
                    seed=seed,
                    directional=directional,
                    implementation_hash=implementation_hash,
                    gpu_load_attestation=normalized_gpu_attestation,
                    gpu_load_attestation_binding=gpu_attestation_binding,
                )
            if trained_now:
                trained += 1
            else:
                resumed += 1
            agents.append(record)
    return {
        "status": "TRAINING_COMPLETE_FOR_REQUESTED_FAMILY",
        "family": family,
        "requested_agents": len(strategies)
        * len(tuple(config["training"]["paired_root_seeds"])),
        "newly_trained_agents": trained,
        "resumed_agents": resumed,
        "agent_records": len(agents),
        "gpu_load_attestation_binding": gpu_attestation_binding,
        "serial_release_witness": serial_release_witness,
        "wall_time_seconds": time.perf_counter() - started,
    }


def _all_agent_records(
    config: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, list[tuple[Any, dict[str, Any]]]], str]:
    implementation_hash = implementation_sha256()
    result: dict[str, list[tuple[Any, dict[str, Any]]]] = {"mf": [], "nmf": []}
    for strategy in ("mf", "nmf"):
        directional = _strategy_config(config, strategy)
        for seed in directional.training_seeds:
            loaded = _load_agent_record(
                output_dir=output_dir,
                strategy=strategy,
                seed=seed,
                directional=directional,
                config=config,
                implementation_hash=implementation_hash,
                replay_validation_candidates=True,
            )
            if loaded is None:
                raise FileNotFoundError(f"missing {strategy} seed {seed} checkpoint")
            result[strategy].append(loaded)
    return result, implementation_hash


def _sentinel_terminal_fingerprint(
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = terminal.get("validation_candidate_states")
    if not isinstance(candidates, list) or not candidates:
        raise CorruptAgentArtifactError("sentinel terminal candidates are missing")
    candidate_rows: list[dict[str, Any]] = []
    for row in candidates:
        state_hash = state_dict_sha256(row["model_state"])
        if state_hash != row.get("state_sha256"):
            raise CorruptAgentArtifactError(
                "sentinel terminal candidate tensor/hash mismatch"
            )
        candidate_rows.append(
            {
                "epoch": int(row["epoch"]),
                "state_sha256": state_hash,
                "lineage_sha256": row["lineage_sha256"],
            }
        )
    terminal_model_hash = state_dict_sha256(terminal["terminal_model_state"])
    selected_hash = state_dict_sha256(terminal["validation_selected_state"])
    if (
        terminal_model_hash != terminal.get("terminal_model_sha256")
        or selected_hash != terminal.get("validation_selected_sha256")
    ):
        raise CorruptAgentArtifactError("sentinel terminal endpoint hash mismatch")
    return {
        "terminal_epoch": int(terminal["terminal_epoch"]),
        "terminal_model_sha256": terminal_model_hash,
        "validation_selected_sha256": selected_hash,
        "validation_candidates": candidate_rows,
        "optimizer_state_sha256": _training_state_sha256(
            _optimizer_state_cpu(terminal["optimizer_state"])
        ),
        "rng_state_sha256": _training_state_sha256(
            _terminal_rng_snapshot(terminal)
        ),
        "score_baseline": float(terminal["score_baseline"]),
        "training_lineage_schema_version": terminal[
            "training_lineage_schema_version"
        ],
        "training_lineage_genesis_sha256": terminal[
            "training_lineage_genesis_sha256"
        ],
        "training_lineage_terminal_sha256": terminal[
            "training_lineage_terminal_sha256"
        ],
        "agent_local_rng_contract": terminal["agent_local_rng_contract"],
    }


def _sentinel_training_evidence(
    training_record: Mapping[str, Any], terminal: Mapping[str, Any]
) -> dict[str, Any]:
    training_payload = copy.deepcopy(dict(training_record))
    training_payload.pop("wall_time_seconds", None)
    return {
        "training_record_without_wall_time": training_payload,
        "terminal_fingerprint": _sentinel_terminal_fingerprint(terminal),
    }


def _sentinel_retrain_audit_valid(
    audit: Any,
    *,
    expected_epochs: int,
    expected_population_agents: int,
    expected_roots: Mapping[str, int] | None = None,
) -> bool:
    try:
        if not isinstance(audit, Mapping) or set(audit) != {
            "schema_version",
            "claim_scope",
            "sentinel_rule",
            "execution_mode",
            "sentinel_agent_count",
            "population_agent_count",
            "families",
            "all_sentinels_exactly_reproduced",
            "temporary_artifacts_deleted",
            "replay_ledger_sha256",
        }:
            return False
        families = audit["families"]
        if (
            audit["schema_version"] != "t9.1.3-family-sentinel-retrain-v1"
            or audit["sentinel_rule"] != "first_frozen_paired_root_per_family"
            or audit["execution_mode"]
            != "sequential_same_runtime_agent_local_rng_isolation"
            or int(audit["sentinel_agent_count"]) != 2
            or int(audit["population_agent_count"]) != expected_population_agents
            or not isinstance(families, list)
            or len(families) != 2
            or audit["all_sentinels_exactly_reproduced"] is not True
            or audit["temporary_artifacts_deleted"] is not True
            or audit["replay_ledger_sha256"] != _canonical_sha256(families)
        ):
            return False
        expected_scope = (
            "two_of_forty_production_agents"
            if expected_population_agents == 40
            else f"two_of_{expected_population_agents}_pilot_agents"
        )
        if audit["claim_scope"] != expected_scope:
            return False
        seen: set[str] = set()
        for row in families:
            if not isinstance(row, Mapping) or set(row) != {
                "strategy",
                "root_seed",
                "epochs_retrained",
                "validation_candidates_retrained",
                "training_evidence_sha256",
                "epoch_lineage_ledger_sha256",
                "terminal_model_sha256",
                "terminal_optimizer_sha256",
                "terminal_rng_sha256",
                "validation_selected_sha256",
                "maximum_numeric_absolute_error",
                "all_epoch_metrics_candidates_and_terminal_exact",
            }:
                return False
            strategy = str(row["strategy"])
            seen.add(strategy)
            if (
                strategy not in {"mf", "nmf"}
                or int(row["epochs_retrained"]) != expected_epochs
                or int(row["validation_candidates_retrained"]) < 2
                or not all(
                    _is_lower_sha256(row[name])
                    for name in (
                        "training_evidence_sha256",
                        "epoch_lineage_ledger_sha256",
                        "terminal_model_sha256",
                        "terminal_optimizer_sha256",
                        "terminal_rng_sha256",
                        "validation_selected_sha256",
                    )
                )
                or float(row["maximum_numeric_absolute_error"]) != 0.0
                or row["all_epoch_metrics_candidates_and_terminal_exact"]
                is not True
                or (
                    expected_roots is not None
                    and int(row["root_seed"]) != int(expected_roots[strategy])
                )
            ):
                return False
        return seen == {"mf", "nmf"}
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _sentinel_full_retrain_audit(
    config: Mapping[str, Any],
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]],
    *,
    scratch_root: str | Path,
) -> dict[str, Any]:
    """Fully retrain one preregistered root per family and compare exact evidence."""

    th = _require_torch()
    roots = tuple(int(value) for value in config["training"]["paired_root_seeds"])
    if not roots:
        raise ValueError("sentinel retrain requires a frozen paired root")
    sentinel_seed = roots[0]
    population_count = sum(len(agents[strategy]) for strategy in ("mf", "nmf"))
    scratch_parent = Path(scratch_root)
    scratch_parent_existed = scratch_parent.exists()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    ambient_rng = _training_rng_snapshot()
    family_rows: list[dict[str, Any]] = []
    temporary_paths: list[Path] = []
    try:
        for strategy in ("mf", "nmf"):
            matches = [
                record
                for _, record in agents[strategy]
                if int(record["root_seed"]) == sentinel_seed
            ]
            if len(matches) != 1:
                raise CorruptAgentArtifactError(
                    f"sentinel root census mismatch for {strategy} seed {sentinel_seed}"
                )
            original_record = matches[0]
            original_terminal_path = _resolve_repo_path(
                original_record["terminal_path"]
            )
            try:
                original_terminal = th.load(
                    original_terminal_path, map_location="cpu", weights_only=False
                )
            except (OSError, EOFError, RuntimeError, ValueError, TypeError) as error:
                raise CorruptAgentArtifactError(
                    f"sentinel original terminal cannot load for {strategy}"
                ) from error
            directional = _strategy_config(config, strategy)
            replay_model, replay_training, replay_terminal = _train_paper_agent(
                strategy, sentinel_seed, directional
            )
            with tempfile.TemporaryDirectory(
                prefix=f"{strategy}_seed_{sentinel_seed}_", dir=scratch_parent
            ) as temporary:
                temporary_path = Path(temporary)
                temporary_paths.append(temporary_path)
                replay_terminal_path = temporary_path / "sentinel.terminal.pt"
                replay_training_path = temporary_path / "sentinel.training.json"
                _atomic_torch(replay_terminal, replay_terminal_path)
                _atomic_json(
                    {"training_record": replay_training}, replay_training_path
                )
                replay_terminal = th.load(
                    replay_terminal_path, map_location="cpu", weights_only=False
                )
                replay_training = _load_json(replay_training_path)[
                    "training_record"
                ]
                _validate_validation_evidence(
                    training_record=replay_training,
                    terminal=replay_terminal,
                    model=replay_model,
                    directional=directional,
                    strategy=strategy,
                    seed=sentinel_seed,
                    replay_physics=False,
                )
                replay_lineage = _validate_training_lineage(
                    training_record=replay_training,
                    terminal=replay_terminal,
                    model=replay_model,
                    directional=directional,
                    strategy=strategy,
                    seed=sentinel_seed,
                )
                original_evidence = _sentinel_training_evidence(
                    original_record["training_record"], original_terminal
                )
                replay_evidence = _sentinel_training_evidence(
                    replay_training, replay_terminal
                )
                maximum_error = _assert_nested_replay_match(
                    original_evidence,
                    replay_evidence,
                    label=f"sentinel retrain {strategy} seed {sentinel_seed}",
                )
                original_digest = _canonical_sha256(original_evidence)
                replay_digest = _canonical_sha256(replay_evidence)
                if original_digest != replay_digest or maximum_error != 0.0:
                    raise CorruptAgentArtifactError(
                        f"sentinel retrain evidence mismatch for {strategy} "
                        f"seed {sentinel_seed}"
                    )
                terminal_fingerprint = replay_evidence["terminal_fingerprint"]
                family_rows.append(
                    {
                        "strategy": strategy,
                        "root_seed": sentinel_seed,
                        "epochs_retrained": int(directional.train_epochs),
                        "validation_candidates_retrained": len(
                            replay_terminal["validation_candidate_states"]
                        ),
                        "training_evidence_sha256": replay_digest,
                        "epoch_lineage_ledger_sha256": replay_lineage[
                            "chain_ledger_sha256"
                        ],
                        "terminal_model_sha256": terminal_fingerprint[
                            "terminal_model_sha256"
                        ],
                        "terminal_optimizer_sha256": terminal_fingerprint[
                            "optimizer_state_sha256"
                        ],
                        "terminal_rng_sha256": terminal_fingerprint[
                            "rng_state_sha256"
                        ],
                        "validation_selected_sha256": terminal_fingerprint[
                            "validation_selected_sha256"
                        ],
                        "maximum_numeric_absolute_error": 0.0,
                        "all_epoch_metrics_candidates_and_terminal_exact": True,
                    }
                )
    finally:
        th.set_rng_state(ambient_rng["torch_cpu"])
        if th.cuda.is_available():
            th.cuda.set_rng_state_all(ambient_rng["torch_cuda"])
        np.random.set_state(ambient_rng["numpy"])
        if not scratch_parent_existed:
            try:
                scratch_parent.rmdir()
            except OSError:
                pass
    temporary_deleted = all(not path.exists() for path in temporary_paths)
    audit = {
        "schema_version": "t9.1.3-family-sentinel-retrain-v1",
        "claim_scope": (
            "two_of_forty_production_agents"
            if population_count == 40
            else f"two_of_{population_count}_pilot_agents"
        ),
        "sentinel_rule": "first_frozen_paired_root_per_family",
        "execution_mode": "sequential_same_runtime_agent_local_rng_isolation",
        "sentinel_agent_count": 2,
        "population_agent_count": population_count,
        "families": family_rows,
        "all_sentinels_exactly_reproduced": len(family_rows) == 2,
        "temporary_artifacts_deleted": temporary_deleted,
        "replay_ledger_sha256": _canonical_sha256(family_rows),
    }
    if not _sentinel_retrain_audit_valid(
        audit,
        expected_epochs=int(config["training"]["epochs"]),
        expected_population_agents=population_count,
        expected_roots={"mf": sentinel_seed, "nmf": sentinel_seed},
    ):
        raise CorruptAgentArtifactError("sentinel retrain audit failed closed")
    return audit


def _select_agents(
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]]
) -> tuple[dict[str, tuple[Any, Mapping[str, Any]]], list[dict[str, Any]]]:
    selected: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    ledger: list[dict[str, Any]] = []
    for strategy in ("mf", "nmf"):
        candidates = agents[strategy]
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]["training_record"]["best_validation_score"]),
                int(item[1]["training_record"]["best_validation_epoch"]),
                int(item[1]["root_seed"]),
            ),
        )
        selected[strategy] = ranked[0]
        selected_seed = int(ranked[0][1]["root_seed"])
        for _, record in candidates:
            train_record = record["training_record"]
            for snapshot in train_record["validation_history"]:
                ledger.append(
                    {
                        "strategy": strategy,
                        "root_seed": int(record["root_seed"]),
                        "checkpoint_epoch": int(snapshot["epoch"]),
                        "selection_split": "validation_only",
                        "selection_score": float(snapshot["selection_score"]),
                        "trajectory_count": int(
                            len(train_record["validation_seeds_used_for_checkpoint_selection_only"])
                            * int(record["strategy_config"]["validation_batch_size"])
                        ),
                        "eligible_within_agent": True,
                        "selected_within_agent": int(snapshot["epoch"])
                        == int(train_record["best_validation_epoch"]),
                        "selected_across_agents": (
                            int(record["root_seed"]) == selected_seed
                            and int(snapshot["epoch"])
                            == int(train_record["best_validation_epoch"])
                        ),
                        "test_or_formal_used": False,
                        "checkpoint_tensor_sha256": snapshot[
                            "candidate_state_sha256"
                        ],
                        "training_lineage_sha256": snapshot[
                            "training_lineage_sha256"
                        ],
                        "status": "COMPLETE",
                    }
                )
    return selected, ledger


def _selection_summary(
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]],
    selected: Mapping[str, tuple[Any, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Expose selection multiplicity and validation gains without a performance claim."""

    summary: dict[str, Any] = {}
    warning = (
        "best-of-20 agents and best-of-validation-checkpoints can be optimistic; "
        "these validation scores are selection diagnostics, not an unbiased performance "
        "estimate or a T9.1.3 performance gate; six-state qualification seeds are held out"
    )
    for strategy in ("mf", "nmf"):
        population = list(agents[strategy])
        records = [item[1]["training_record"] for item in population]
        candidate_counts = [len(record["validation_history"]) for record in records]
        if not population or len(set(candidate_counts)) != 1:
            raise CorruptAgentArtifactError(
                f"{strategy} selection candidate multiplicity is incomplete"
            )
        winner = selected[strategy][1]
        summary[strategy] = {
            "selected_root_seed": int(winner["root_seed"]),
            "selected_checkpoint_epoch": int(
                winner["training_record"]["best_validation_epoch"]
            ),
            "selected_validation_score": float(
                winner["training_record"]["best_validation_score"]
            ),
            "checkpoint_sha256": winner["canonical_tensor_sha256"],
            "agent_count": len(population),
            "validation_candidates_per_agent": candidate_counts[0],
            "total_agent_checkpoint_candidates": sum(candidate_counts),
            "all_agent_best_validation_score_distribution": _distribution(
                record["best_validation_score"] for record in records
            ),
            "all_agent_initial_validation_score_distribution": _distribution(
                record["initial_validation_score"] for record in records
            ),
            "all_agent_initial_to_best_gain_distribution": _distribution(
                float(record["best_validation_score"])
                - float(record["initial_validation_score"])
                for record in records
            ),
            "all_agent_selected_checkpoint_epoch_distribution": _distribution(
                record["best_validation_epoch"] for record in records
            ),
            "selection_optimism_warning": warning,
        }
    return summary


def _logical_state_vector(label: str) -> np.ndarray:
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    vectors = {
        "x_plus": inverse_sqrt_two * np.asarray([1.0, 1.0], dtype=np.complex128),
        "x_minus": inverse_sqrt_two * np.asarray([1.0, -1.0], dtype=np.complex128),
        "y_plus": inverse_sqrt_two * np.asarray([1.0, 1.0j], dtype=np.complex128),
        "y_minus": inverse_sqrt_two * np.asarray([1.0, -1.0j], dtype=np.complex128),
        "z_plus": np.asarray([1.0, 0.0], dtype=np.complex128),
        "z_minus": np.asarray([0.0, 1.0], dtype=np.complex128),
    }
    if label not in vectors:
        raise ValueError(f"unknown logical state label: {label}")
    vector = vectors[label]
    density = vector[:, None] * vector.conj()[None, :]
    if not np.allclose(
        density, logical_eigenstate_density(label), rtol=0.0, atol=2.0e-16
    ):
        raise RuntimeError(f"explicit logical vector drifted for {label}")
    return vector.copy()


def _projected_logical_metrics(
    projected_raw: np.ndarray, label: str
) -> dict[str, Any]:
    matrix = np.asarray(projected_raw, dtype=np.complex128)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("projected logical density must be a finite 2x2 matrix")
    hermiticity_error = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    if hermiticity_error > 2.0e-9:
        raise ValueError("projected logical density is not Hermitian")
    logical = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(logical)
    if float(np.min(eigenvalues)) < -2.0e-8:
        raise ValueError("projected logical density is not positive semidefinite")
    pauli = np.asarray(
        [np.trace(operator @ logical).real for operator in PAULIS],
        dtype=np.float64,
    )
    axis, expected_eigenvalue = STATE_SPEC[label]
    axis_index = ("I", "X", "Y", "Z").index(axis)
    target = logical_eigenstate_density(label)
    fidelity = float(np.trace(target @ logical).real)
    survival = float(pauli[0])
    signed_axis = float(expected_eigenvalue * pauli[axis_index])
    if not np.isclose(
        2.0 * fidelity, survival + signed_axis, rtol=0.0, atol=2.0e-13
    ):
        raise ValueError("projected logical Pauli/fidelity identity failed")
    return {
        "logical": logical,
        "hermiticity_error": hermiticity_error,
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "code_survival": survival,
        "code_weighted_I": float(pauli[0]),
        "code_weighted_X": float(pauli[1]),
        "code_weighted_Y": float(pauli[2]),
        "code_weighted_Z": float(pauli[3]),
        "state_fidelity_unconditional": fidelity,
        "signed_axis_signal": signed_axis,
        "pauli_axis": axis,
        "expected_eigenvalue": float(expected_eigenvalue),
    }


def _set_logical_state(
    simulator: PaperCausalSBSTrajectorySimulator, label: str
) -> None:
    """Install one of six logical states without modifying the historical backend."""

    th = _require_torch()
    logical_vector = _logical_state_vector(label)
    coefficients = th.as_tensor(
        logical_vector,
        dtype=simulator.complex_dtype,
        device=simulator.device,
    )
    physical = simulator.logical_isometry @ coefficients
    physical = physical / th.linalg.vector_norm(physical)
    simulator.initial_state_vector = physical
    simulator.target_cavity_density = physical[:, None] * physical.conj()[None, :]


def _evaluate_six_state(
    config: Mapping[str, Any],
    selected: Mapping[str, tuple[Any, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run a non-promotional six-state stochastic qualification with raw rows."""

    th = _require_torch()
    training = config["training"]
    qualification = config["six_state_qualification"]
    trajectory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    primary_cutoff = int(training["cutoff"])
    confirmation_cutoff = int(training["confirmation_cutoff"])
    strata = (
        (
            f"primary_cutoff{primary_cutoff}",
            primary_cutoff,
            tuple(int(seed) for seed in qualification["seeds"]),
            int(qualification["batch_size"]),
        ),
        (
            f"selected_confirmation_cutoff{confirmation_cutoff}",
            confirmation_cutoff,
            tuple(int(seed) for seed in qualification["confirmation_seeds"]),
            int(qualification["confirmation_batch_size"]),
        ),
    )
    reconstruction: dict[str, Any] = {}
    for stratum, cutoff, seeds, batch_size in strata:
        reconstruction[stratum] = {}
        for strategy in ("standard", "mf", "nmf"):
            model = None if strategy == "standard" else selected[strategy][0]
            agent_seed = None if strategy == "standard" else int(selected[strategy][1]["root_seed"])
            state_outputs: dict[str, list[np.ndarray]] = {label: [] for label in STATE_LABELS}
            max_trace_error = 0.0
            max_hermiticity_error = 0.0
            minimum_eigenvalue = 1.0
            max_projected_hermiticity_error = 0.0
            raw_codeword_overlap: complex | None = None
            raw_gram_condition_number: float | None = None
            for label in STATE_LABELS:
                axis, expected_eigenvalue = STATE_SPEC[label]
                for seed in seeds:
                    simulator = PaperCausalSBSTrajectorySimulator(
                        PaperCausalSBSConfig(
                            cutoff=cutoff,
                            full_cycles=int(qualification["full_cycles"]),
                            batch_size=batch_size,
                            projector_delta=float(training["projector_delta"]),
                            cavity_lifetime_us=float(
                                training["noise_parameters_us"]["cavity_lifetime"]
                            ),
                            ancilla_t1_us=float(training["noise_parameters_us"]["ancilla_t1"]),
                            ancilla_t2_us=float(training["noise_parameters_us"]["ancilla_t2"]),
                            device=str(training["device"]),
                            real_dtype=str(training["real_dtype"]),
                        )
                    )
                    _set_logical_state(simulator, label)
                    with th.no_grad():
                        result = simulator.run(
                            control_policy=model,
                            seed=int(seed),
                            record_cycle_metrics=False,
                        )
                    if result.feedback_timeline != PAPER_CAUSAL_FEEDBACK_TIMELINE:
                        raise RuntimeError("six-state evaluator did not use the frozen paper-causal timeline")
                    current_overlap = simulator.raw_logical_codeword_overlap
                    current_condition = simulator.raw_logical_gram_condition_number
                    if raw_codeword_overlap is None:
                        raw_codeword_overlap = current_overlap
                        raw_gram_condition_number = current_condition
                    elif not (
                        np.isclose(raw_codeword_overlap.real, current_overlap.real, rtol=0.0, atol=1.0e-14)
                        and np.isclose(raw_codeword_overlap.imag, current_overlap.imag, rtol=0.0, atol=1.0e-14)
                        and np.isclose(raw_gram_condition_number, current_condition, rtol=0.0, atol=1.0e-14)
                    ):
                        raise RuntimeError("finite-cutoff logical-basis diagnostics drifted within a stratum")
                    final = result.final_cavity_density.detach().cpu().numpy()
                    isometry = simulator.logical_isometry.detach().cpu().numpy()
                    projected = np.einsum(
                        "ai,ban,nj->bij",
                        isometry.conj(),
                        final,
                        isometry,
                        optimize=True,
                    )
                    # The explicit einsum above uses named physical indices only after
                    # transposition; retain a direct expression as an independent shape guard.
                    projected_direct = np.stack(
                        [isometry.conj().T @ rho @ isometry for rho in final], axis=0
                    )
                    if not np.allclose(projected, projected_direct, rtol=0.0, atol=1.0e-12):
                        raise RuntimeError(
                            "logical projection einsum/direct implementations disagree"
                        )
                    outcomes = result.outcomes.detach().cpu().numpy()
                    probabilities = result.conditional_probabilities.detach().cpu().numpy()
                    decisions = (
                        result.decision_physical_controls.detach().cpu().numpy()
                    )
                    controls = decisions[:, :-1, :]
                    post_controls = decisions[:, 1:, :]
                    terminal_rotation = (
                        result.terminal_virtual_rotation.detach().cpu().numpy()
                    )
                    if not np.array_equal(
                        terminal_rotation, post_controls[:, -1, 14]
                    ):
                        raise RuntimeError("terminal feedback control is missing or inconsistent")
                    if not np.array_equal(post_controls[:, :-1, :], controls[:, 1:, :]):
                        raise RuntimeError("pending action was not reused by the next half-cycle")
                    max_trace_error = max(max_trace_error, result.maximum_trace_error)
                    max_hermiticity_error = max(
                        max_hermiticity_error, result.maximum_hermiticity_error
                    )
                    minimum_eigenvalue = min(minimum_eigenvalue, result.minimum_final_eigenvalue)
                    for member in range(batch_size):
                        projected_hermiticity_error = float(
                            np.linalg.norm(
                                projected_direct[member]
                                - projected_direct[member].conj().T,
                                ord="fro",
                            )
                        )
                        max_projected_hermiticity_error = max(
                            max_projected_hermiticity_error,
                            projected_hermiticity_error,
                        )
                        if projected_hermiticity_error > 2.0e-9:
                            raise RuntimeError(
                                "projected logical density is not Hermitian before stabilization"
                            )
                        logical_metrics = _projected_logical_metrics(
                            projected_direct[member], label
                        )
                        logical = logical_metrics["logical"]
                        state_outputs[label].append(logical)
                        trajectory_id = (
                            f"{stratum}:{strategy}:{label}:{seed}:{member}"
                        )
                        trajectory_rows.append(
                            {
                                "trajectory_id": trajectory_id,
                                "stratum": stratum,
                                "strategy": strategy,
                                "selected_agent_seed": agent_seed,
                                "initial_state": label,
                                "pauli_axis": axis,
                                "expected_eigenvalue": expected_eigenvalue,
                                "evaluation_seed": int(seed),
                                "evaluation_cluster_id": f"{stratum}:seed_{seed}",
                                "rng_stream_id": f"torch_cpu_seed_{seed}:member_{member}",
                                "batch_member": member,
                                "state_definition_id": qualification["state_definition_id"],
                                "logical_operator_definition_id": qualification[
                                    "logical_operator_definition_id"
                                ],
                                "raw_codeword_overlap_real": float(current_overlap.real),
                                "raw_codeword_overlap_imag": float(current_overlap.imag),
                                "raw_logical_gram_condition_number": float(current_condition),
                                "cutoff": cutoff,
                                "full_cycles": int(qualification["full_cycles"]),
                                "half_cycles": 2 * int(qualification["full_cycles"]),
                                "timing_profile_id": simulator.config.timing.profile_id,
                                "simulated_time_us": (
                                    simulator.config.timing.full_cycle_duration_ns
                                    * int(qualification["full_cycles"])
                                    / 1000.0
                                ),
                                "projected_rho_00_real": float(logical[0, 0].real),
                                "projected_rho_00_imag": float(logical[0, 0].imag),
                                "projected_rho_01_real": float(logical[0, 1].real),
                                "projected_rho_01_imag": float(logical[0, 1].imag),
                                "projected_rho_10_real": float(logical[1, 0].real),
                                "projected_rho_10_imag": float(logical[1, 0].imag),
                                "projected_rho_11_real": float(logical[1, 1].real),
                                "projected_rho_11_imag": float(logical[1, 1].imag),
                                "joint_trace_error_max_for_batch": float(
                                    result.maximum_trace_error
                                ),
                                "joint_hermiticity_error_max_for_batch": float(
                                    result.maximum_hermiticity_error
                                ),
                                "minimum_final_joint_eigenvalue_for_batch": float(
                                    result.minimum_final_eigenvalue
                                ),
                                "projected_hermiticity_error_before_stabilization": (
                                    projected_hermiticity_error
                                ),
                                "code_weighted_I": logical_metrics["code_weighted_I"],
                                "code_weighted_X": logical_metrics["code_weighted_X"],
                                "code_weighted_Y": logical_metrics["code_weighted_Y"],
                                "code_weighted_Z": logical_metrics["code_weighted_Z"],
                                "signed_axis_signal": logical_metrics[
                                    "signed_axis_signal"
                                ],
                                "state_fidelity_unconditional": logical_metrics[
                                    "state_fidelity_unconditional"
                                ],
                                "code_survival": logical_metrics["code_survival"],
                                "postselected": False,
                                "included_in_denominator": True,
                                "development_only": True,
                                "censored": False,
                                "status": "COMPLETE",
                            }
                        )
                        for half_index in range(outcomes.shape[1]):
                            project_action = controls[member, half_index]
                            post_action = post_controls[member, half_index]
                            gqf_action = project_to_gqf_action(project_action)
                            post_gqf_action = project_to_gqf_action(post_action)
                            terminal = half_index + 1 == outcomes.shape[1]
                            event_rows.append(
                                {
                                    "trajectory_id": trajectory_id,
                                    "half_cycle": half_index + 1,
                                    "full_cycle_coordinate": (half_index + 1) / 2.0,
                                    "layer_control_decision_prefix_length": half_index,
                                    "physical_layer_application_half_cycle": half_index + 1,
                                    "measurement_outcome_half_cycle": half_index + 1,
                                    "post_measurement_control_decision_prefix_length": half_index + 1,
                                    "post_measurement_virtual_rotation_after_half_cycle": half_index + 1,
                                    "post_measurement_control_layer_application_half_cycle": (
                                        None if terminal else half_index + 2
                                    ),
                                    "uses_current_measurement_outcome_for_post_measurement_control": True,
                                    "terminal_feedback_correction": terminal,
                                    "outcome_g0_e1": int(outcomes[member, half_index]),
                                    "selected_branch_probability": float(
                                        probabilities[member, half_index]
                                    ),
                                    "project_action_order_json": json.dumps(
                                        project_action.tolist(), separators=(",", ":")
                                    ),
                                    "gqf_action_order_json": json.dumps(
                                        gqf_action.tolist(), separators=(",", ":")
                                    ),
                                    "project_action_sha256": hashlib.sha256(
                                        np.ascontiguousarray(project_action).tobytes()
                                    ).hexdigest(),
                                    "post_measurement_project_action_order_json": json.dumps(
                                        post_action.tolist(), separators=(",", ":")
                                    ),
                                    "post_measurement_gqf_action_order_json": json.dumps(
                                        post_gqf_action.tolist(), separators=(",", ":")
                                    ),
                                    "post_measurement_project_action_sha256": hashlib.sha256(
                                        np.ascontiguousarray(post_action).tobytes()
                                    ).hexdigest(),
                                    "evaluation_cluster_id": f"{stratum}:seed_{seed}",
                                    "rng_stream_id": f"torch_cpu_seed_{seed}:member_{member}",
                                    "postselected": False,
                                    "included_in_denominator": True,
                                    "development_only": True,
                                    "status": "COMPLETE",
                                }
                            )
            averaged = {
                label: np.mean(np.stack(values, axis=0), axis=0)
                for label, values in state_outputs.items()
            }
            tomography = reconstruct_code_subchannel(averaged)
            reconstruction[stratum][strategy] = {
                "selected_agent_seed": agent_seed,
                "trajectory_count": sum(len(values) for values in state_outputs.values()),
                "empirical_linear_inversion_subchannel_diagnostic": tomography.to_dict(),
                "monte_carlo_pair_sum_not_deterministic_tolerance": True,
                "empirical_cptni_passed_is_not_a_hard_gate": True,
                "state_definition_id": qualification["state_definition_id"],
                "logical_operator_definition_id": qualification[
                    "logical_operator_definition_id"
                ],
                "monte_carlo_design": qualification["monte_carlo_design"],
                "raw_codeword_overlap_real": float(raw_codeword_overlap.real),
                "raw_codeword_overlap_imag": float(raw_codeword_overlap.imag),
                "raw_logical_gram_condition_number": float(
                    raw_gram_condition_number
                ),
                "diagnostic_scope": (
                    "unconstrained linear inversion of paired-common-random-number six-state "
                    "means clustered by stratum/evaluation seed; retain CP/TNI diagnostics but do not treat finite-sample "
                    "violations as physical-channel proof"
                ),
                "maximum_joint_trace_error": max_trace_error,
                "maximum_joint_hermiticity_error": max_hermiticity_error,
                "maximum_projected_hermiticity_error_before_stabilization": (
                    max_projected_hermiticity_error
                ),
                "minimum_final_joint_eigenvalue": minimum_eigenvalue,
                "paper_lifetime_value": None,
                "paper_lifetime_state": PAPER_NUMERIC_STATE,
            }
    reconstructed_from_rows = _reconstruct_qualification_from_rows(
        config, trajectory_rows
    )
    if _canonical_sha256(reconstruction) != _canonical_sha256(
        reconstructed_from_rows
    ):
        raise RuntimeError(
            "in-memory and raw-row six-state reconstructions disagree"
        )
    return trajectory_rows, event_rows, reconstructed_from_rows


def _reconstruct_qualification_from_rows(
    config: Mapping[str, Any], trajectory_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Derive every published six-state diagnostic from raw trajectory rows."""

    training = config["training"]
    qualification = config["six_state_qualification"]
    strata = (
        f"primary_cutoff{int(training['cutoff'])}",
        f"selected_confirmation_cutoff{int(training['confirmation_cutoff'])}",
    )
    result: dict[str, Any] = {}
    for stratum in strata:
        result[stratum] = {}
        for strategy in ("standard", "mf", "nmf"):
            grouped = [
                row
                for row in trajectory_rows
                if row["stratum"] == stratum and row["strategy"] == strategy
            ]
            if not grouped:
                raise ValueError(
                    f"missing six-state rows for {stratum}/{strategy}"
                )
            averaged: dict[str, np.ndarray] = {}
            for label in STATE_LABELS:
                state_rows = [row for row in grouped if row["initial_state"] == label]
                if not state_rows:
                    raise ValueError(
                        f"missing logical state {label} for {stratum}/{strategy}"
                    )
                matrices = np.asarray(
                    [
                        [
                            [
                                complex(
                                    float(row["projected_rho_00_real"]),
                                    float(row["projected_rho_00_imag"]),
                                ),
                                complex(
                                    float(row["projected_rho_01_real"]),
                                    float(row["projected_rho_01_imag"]),
                                ),
                            ],
                            [
                                complex(
                                    float(row["projected_rho_10_real"]),
                                    float(row["projected_rho_10_imag"]),
                                ),
                                complex(
                                    float(row["projected_rho_11_real"]),
                                    float(row["projected_rho_11_imag"]),
                                ),
                            ],
                        ]
                        for row in state_rows
                    ],
                    dtype=np.complex128,
                )
                if not np.all(np.isfinite(matrices)):
                    raise ValueError("six-state projected matrices must be finite")
                averaged[label] = np.mean(matrices, axis=0)
            selected_seeds = {row["selected_agent_seed"] for row in grouped}
            if len(selected_seeds) != 1:
                raise ValueError("selected agent seed drifted within a method/stratum")
            overlap_real = {float(row["raw_codeword_overlap_real"]) for row in grouped}
            overlap_imag = {float(row["raw_codeword_overlap_imag"]) for row in grouped}
            conditions = {
                float(row["raw_logical_gram_condition_number"])
                for row in grouped
            }
            if not (
                len(overlap_real) == len(overlap_imag) == len(conditions) == 1
            ):
                raise ValueError("logical-basis diagnostics drifted within a stratum")
            tomography = reconstruct_code_subchannel(averaged)
            result[stratum][strategy] = {
                "selected_agent_seed": next(iter(selected_seeds)),
                "trajectory_count": len(grouped),
                "empirical_linear_inversion_subchannel_diagnostic": (
                    tomography.to_dict()
                ),
                "monte_carlo_pair_sum_not_deterministic_tolerance": True,
                "empirical_cptni_passed_is_not_a_hard_gate": True,
                "state_definition_id": qualification["state_definition_id"],
                "logical_operator_definition_id": qualification[
                    "logical_operator_definition_id"
                ],
                "monte_carlo_design": qualification["monte_carlo_design"],
                "raw_codeword_overlap_real": next(iter(overlap_real)),
                "raw_codeword_overlap_imag": next(iter(overlap_imag)),
                "raw_logical_gram_condition_number": next(iter(conditions)),
                "diagnostic_scope": (
                    "unconstrained linear inversion of paired-common-random-number six-state "
                    "means clustered by stratum/evaluation seed; retain CP/TNI diagnostics but do not treat finite-sample "
                    "violations as physical-channel proof"
                ),
                "maximum_joint_trace_error": max(
                    float(row["joint_trace_error_max_for_batch"])
                    for row in grouped
                ),
                "maximum_joint_hermiticity_error": max(
                    float(row["joint_hermiticity_error_max_for_batch"])
                    for row in grouped
                ),
                "maximum_projected_hermiticity_error_before_stabilization": max(
                    float(
                        row[
                            "projected_hermiticity_error_before_stabilization"
                        ]
                    )
                    for row in grouped
                ),
                "minimum_final_joint_eigenvalue": min(
                    float(row["minimum_final_joint_eigenvalue_for_batch"])
                    for row in grouped
                ),
                "paper_lifetime_value": None,
                "paper_lifetime_state": PAPER_NUMERIC_STATE,
            }
    return result


def _trajectory_rows_integrity(
    config: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    selected_agent_seeds: Mapping[str, int | None],
) -> bool:
    """Recompute every scalar logical metric from each stored 2x2 projection."""

    try:
        training = config["training"]
        qualification = config["six_state_qualification"]
        stratum_specs = {
            f"primary_cutoff{int(training['cutoff'])}": (
                int(training["cutoff"]),
                {int(seed) for seed in qualification["seeds"]},
                int(qualification["batch_size"]),
            ),
            f"selected_confirmation_cutoff{int(training['confirmation_cutoff'])}": (
                int(training["confirmation_cutoff"]),
                {int(seed) for seed in qualification["confirmation_seeds"]},
                int(qualification["confirmation_batch_size"]),
            ),
        }
        for row in trajectory_rows:
            if list(row) != list(TRAJECTORY_ROW_KEYS):
                return False
            stratum = str(row["stratum"])
            if stratum not in stratum_specs:
                return False
            cutoff, seeds, batch_size = stratum_specs[stratum]
            strategy = str(row["strategy"])
            label = str(row["initial_state"])
            evaluation_seed = int(row["evaluation_seed"])
            member = int(row["batch_member"])
            if strategy not in {"standard", "mf", "nmf"} or label not in STATE_SPEC:
                return False
            axis, eigenvalue = STATE_SPEC[label]
            expected_selected = selected_agent_seeds[strategy]
            if (
                row["selected_agent_seed"] != expected_selected
                or row["pauli_axis"] != axis
                or float(row["expected_eigenvalue"]) != float(eigenvalue)
                or evaluation_seed not in seeds
                or member not in range(batch_size)
                or int(row["cutoff"]) != cutoff
                or int(row["full_cycles"]) != int(qualification["full_cycles"])
                or int(row["half_cycles"])
                != 2 * int(qualification["full_cycles"])
                or row["trajectory_id"]
                != f"{stratum}:{strategy}:{label}:{evaluation_seed}:{member}"
                or row["evaluation_cluster_id"]
                != f"{stratum}:seed_{evaluation_seed}"
                or row["rng_stream_id"]
                != f"torch_cpu_seed_{evaluation_seed}:member_{member}"
                or row["state_definition_id"] != qualification["state_definition_id"]
                or row["logical_operator_definition_id"]
                != qualification["logical_operator_definition_id"]
                or row["postselected"] is not False
                or row["included_in_denominator"] is not True
                or row["development_only"] is not True
                or row["censored"] is not False
                or row["status"] != "COMPLETE"
            ):
                return False
            matrix = np.asarray(
                [
                    [
                        complex(row["projected_rho_00_real"], row["projected_rho_00_imag"]),
                        complex(row["projected_rho_01_real"], row["projected_rho_01_imag"]),
                    ],
                    [
                        complex(row["projected_rho_10_real"], row["projected_rho_10_imag"]),
                        complex(row["projected_rho_11_real"], row["projected_rho_11_imag"]),
                    ],
                ],
                dtype=np.complex128,
            )
            metrics = _projected_logical_metrics(matrix, label)
            for name in (
                "code_survival",
                "code_weighted_I",
                "code_weighted_X",
                "code_weighted_Y",
                "code_weighted_Z",
                "state_fidelity_unconditional",
                "signed_axis_signal",
            ):
                if not np.isclose(
                    float(row[name]), float(metrics[name]), rtol=0.0, atol=1.0e-12
                ):
                    return False
            diagnostic_names = (
                "raw_codeword_overlap_real",
                "raw_codeword_overlap_imag",
                "raw_logical_gram_condition_number",
                "simulated_time_us",
                "joint_trace_error_max_for_batch",
                "joint_hermiticity_error_max_for_batch",
                "minimum_final_joint_eigenvalue_for_batch",
                "projected_hermiticity_error_before_stabilization",
            )
            if (
                not all(np.isfinite(float(row[name])) for name in diagnostic_names)
                or float(row["raw_logical_gram_condition_number"]) < 1.0
                or float(row["projected_hermiticity_error_before_stabilization"])
                > 2.0e-9
                or float(row["code_survival"]) < -1.0e-8
                or float(row["code_survival"]) > 1.0 + 1.0e-8
                or float(row["state_fidelity_unconditional"]) < -1.0e-9
                or float(row["state_fidelity_unconditional"])
                > float(row["code_survival"]) + 1.0e-9
                or not isinstance(row["timing_profile_id"], str)
                or not row["timing_profile_id"]
            ):
                return False
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return False
    return True


def _independent_prefix_decisions(
    *,
    strategy: str,
    model: Any | None,
    outcomes: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Recompute d0..dH without the simulator's stateful rollout cache."""

    th = _require_torch()
    values = np.asarray(outcomes, dtype=np.int64)
    if values.ndim != 2 or not np.all((values == 0) | (values == 1)):
        raise ValueError("outcome replay matrix must be binary and rank two")
    batch_size, half_cycles = values.shape
    training = config["training"]
    dtype = _torch_dtype(str(training["real_dtype"]))
    device = str(training["device"])
    nominal = nominal_sbs_parameters(device=device, dtype=dtype)
    bounds = th.full((15,), 2.0, dtype=dtype, device=device)
    bounds[-1] = 1.0
    history = th.as_tensor(values, dtype=th.int64, device=device)
    decisions: list[Any] = []
    with th.no_grad():
        for prefix_length in range(half_cycles + 1):
            if strategy == "standard":
                raw = th.zeros((batch_size, 15), dtype=dtype, device=device)
            else:
                if model is None:
                    raise ValueError("learned replay requires the selected model")
                raw = model.forward(history[:, :prefix_length], prefix_length)
                if not isinstance(raw, th.Tensor):
                    raise TypeError("selected policy forward did not return a tensor")
                raw = raw.to(device=device, dtype=dtype)
            if raw.shape != (batch_size, 15) or not bool(
                th.all(th.isfinite(raw)).detach().cpu()
            ):
                raise ValueError("independent policy replay returned invalid controls")
            decisions.append(nominal[None, :] + bounds[None, :] * th.tanh(raw))
    return th.stack(decisions, dim=1).detach().cpu().numpy()


def _six_state_raw_replay_audit(
    config: Mapping[str, Any],
    selected: Mapping[str, tuple[Any, Mapping[str, Any]]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seeded replay in the captured runtime/backend, not cross-runtime bitwise proof."""

    th = _require_torch()
    training = config["training"]
    qualification = config["six_state_qualification"]
    selected_seeds: dict[str, int | None] = {
        "standard": None,
        "mf": int(selected["mf"][1]["root_seed"]),
        "nmf": int(selected["nmf"][1]["root_seed"]),
    }
    if not _trajectory_rows_integrity(
        config,
        trajectory_rows,
        selected_agent_seeds=selected_seeds,
    ):
        raise ValueError("trajectory rows are not matrix/metric/winner self-consistent")
    if not _event_timeline_valid(
        event_rows,
        trajectory_rows,
        half_cycles=2 * int(qualification["full_cycles"]),
    ):
        raise ValueError("event rows fail the causal/action-bound contract")
    rows_by_group: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in trajectory_rows:
        key = (
            str(row["stratum"]),
            str(row["strategy"]),
            str(row["initial_state"]),
            int(row["evaluation_seed"]),
        )
        rows_by_group.setdefault(key, []).append(row)
    events_by_trajectory: dict[str, list[Mapping[str, Any]]] = {}
    for row in event_rows:
        events_by_trajectory.setdefault(str(row["trajectory_id"]), []).append(row)
    maximum_action_error = 0.0
    maximum_probability_error = 0.0
    maximum_projected_rho_error = 0.0
    replayed_trajectories = 0
    replayed_events = 0
    for (stratum, strategy, label, evaluation_seed), group in sorted(
        rows_by_group.items()
    ):
        ordered = sorted(group, key=lambda row: int(row["batch_member"]))
        if [int(row["batch_member"]) for row in ordered] != list(range(len(ordered))):
            raise ValueError("six-state replay batch-member census drifted")
        cutoff = int(ordered[0]["cutoff"])
        simulator = PaperCausalSBSTrajectorySimulator(
            PaperCausalSBSConfig(
                cutoff=cutoff,
                full_cycles=int(qualification["full_cycles"]),
                batch_size=len(ordered),
                projector_delta=float(training["projector_delta"]),
                cavity_lifetime_us=float(
                    training["noise_parameters_us"]["cavity_lifetime"]
                ),
                ancilla_t1_us=float(training["noise_parameters_us"]["ancilla_t1"]),
                ancilla_t2_us=float(training["noise_parameters_us"]["ancilla_t2"]),
                device=str(training["device"]),
                real_dtype=str(training["real_dtype"]),
            )
        )
        _set_logical_state(simulator, label)
        model = None if strategy == "standard" else selected[strategy][0]
        with th.no_grad():
            replay = simulator.run(
                control_policy=model,
                seed=evaluation_seed,
                record_cycle_metrics=False,
            )
        outcomes = replay.outcomes.detach().cpu().numpy()
        probabilities = replay.conditional_probabilities.detach().cpu().numpy()
        decisions = replay.decision_physical_controls.detach().cpu().numpy()
        independent = _independent_prefix_decisions(
            strategy=strategy,
            model=model,
            outcomes=outcomes,
            config=config,
        )
        action_error = float(np.max(np.abs(decisions - independent)))
        maximum_action_error = max(maximum_action_error, action_error)
        if not np.allclose(decisions, independent, rtol=0.0, atol=1.0e-11):
            raise ValueError("stateful and stateless selected-policy replay disagree")
        final = replay.final_cavity_density.detach().cpu().numpy()
        isometry = simulator.logical_isometry.detach().cpu().numpy()
        expected_overlap = simulator.raw_logical_codeword_overlap
        expected_condition = simulator.raw_logical_gram_condition_number
        expected_time_us = (
            float(replay.resource_profile.simulated_physical_time_ns) / 1000.0
        )
        projected = np.stack(
            [isometry.conj().T @ density @ isometry for density in final], axis=0
        )
        for member, trajectory in enumerate(ordered):
            trajectory_id = str(trajectory["trajectory_id"])
            events = sorted(
                events_by_trajectory.get(trajectory_id, []),
                key=lambda row: int(row["half_cycle"]),
            )
            if len(events) != outcomes.shape[1]:
                raise ValueError("six-state replay event census drifted")
            raw_outcomes = np.asarray(
                [int(row["outcome_g0_e1"]) for row in events], dtype=np.int64
            )
            if not np.array_equal(raw_outcomes, outcomes[member]):
                raise ValueError("seeded physics replay outcome mismatch")
            raw_matrix = np.asarray(
                [
                    [
                        complex(trajectory["projected_rho_00_real"], trajectory["projected_rho_00_imag"]),
                        complex(trajectory["projected_rho_01_real"], trajectory["projected_rho_01_imag"]),
                    ],
                    [
                        complex(trajectory["projected_rho_10_real"], trajectory["projected_rho_10_imag"]),
                        complex(trajectory["projected_rho_11_real"], trajectory["projected_rho_11_imag"]),
                    ],
                ],
                dtype=np.complex128,
            )
            replay_matrix = 0.5 * (projected[member] + projected[member].conj().T)
            rho_error = float(np.max(np.abs(raw_matrix - replay_matrix)))
            maximum_projected_rho_error = max(maximum_projected_rho_error, rho_error)
            if not np.allclose(raw_matrix, replay_matrix, rtol=1.0e-10, atol=5.0e-10):
                raise ValueError("seeded physics replay projected-rho mismatch")
            projected_hermiticity = float(
                np.linalg.norm(projected[member] - projected[member].conj().T, ord="fro")
            )
            if (
                not np.isclose(
                    float(trajectory["raw_codeword_overlap_real"]),
                    float(expected_overlap.real),
                    rtol=0.0,
                    atol=1.0e-14,
                )
                or not np.isclose(
                    float(trajectory["raw_codeword_overlap_imag"]),
                    float(expected_overlap.imag),
                    rtol=0.0,
                    atol=1.0e-14,
                )
                or not np.isclose(
                    float(trajectory["raw_logical_gram_condition_number"]),
                    float(expected_condition),
                    rtol=0.0,
                    atol=1.0e-13,
                )
                or trajectory["timing_profile_id"]
                != replay.resource_profile.timing_profile_id
                or not np.isclose(
                    float(trajectory["simulated_time_us"]),
                    expected_time_us,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.isclose(
                    float(trajectory["projected_hermiticity_error_before_stabilization"]),
                    projected_hermiticity,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.isclose(
                    float(trajectory["joint_trace_error_max_for_batch"]),
                    float(replay.maximum_trace_error),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.isclose(
                    float(trajectory["joint_hermiticity_error_max_for_batch"]),
                    float(replay.maximum_hermiticity_error),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                or not np.isclose(
                    float(trajectory["minimum_final_joint_eigenvalue_for_batch"]),
                    float(replay.minimum_final_eigenvalue),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise ValueError("seeded physics replay diagnostic mismatch")
            for half_index, event in enumerate(events):
                project_action = np.asarray(
                    json.loads(str(event["project_action_order_json"])),
                    dtype=np.float64,
                )
                post_action = np.asarray(
                    json.loads(
                        str(event["post_measurement_project_action_order_json"])
                    ),
                    dtype=np.float64,
                )
                probability_error = abs(
                    float(event["selected_branch_probability"])
                    - float(probabilities[member, half_index])
                )
                maximum_probability_error = max(
                    maximum_probability_error, probability_error
                )
                maximum_action_error = max(
                    maximum_action_error,
                    float(
                        np.max(
                            np.abs(project_action - independent[member, half_index])
                        )
                    ),
                    float(
                        np.max(
                            np.abs(post_action - independent[member, half_index + 1])
                        )
                    ),
                )
                if (
                    not np.allclose(
                        project_action,
                        independent[member, half_index],
                        rtol=0.0,
                        atol=1.0e-11,
                    )
                    or not np.allclose(
                        post_action,
                        independent[member, half_index + 1],
                        rtol=0.0,
                        atol=1.0e-11,
                    )
                    or not np.isclose(
                        float(event["selected_branch_probability"]),
                        float(probabilities[member, half_index]),
                        rtol=1.0e-10,
                        atol=1.0e-11,
                    )
                ):
                    raise ValueError("selected checkpoint/action/probability replay mismatch")
            replayed_trajectories += 1
            replayed_events += len(events)
    if replayed_trajectories != len(trajectory_rows) or replayed_events != len(event_rows):
        raise ValueError("six-state replay did not consume the full denominator")
    return {
        "schema_version": "t9.1.3-six-state-same-backend-replay-v1",
        "scope": (
            "seeded same-runtime/same-backend replay of selected checkpoints, RNG, "
            "physics, actions and raw rows; not cross-runtime bitwise reproducibility "
            "and not an independent second physics backend"
        ),
        "group_count": len(rows_by_group),
        "trajectory_count": replayed_trajectories,
        "event_count": replayed_events,
        "maximum_action_absolute_error": maximum_action_error,
        "maximum_branch_probability_absolute_error": maximum_probability_error,
        "maximum_projected_rho_absolute_error": maximum_projected_rho_error,
        "all_rows_replayed": True,
    }


def _source_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    prereg = root / "configs/gqf_official/paper_exact_preregistration.json"
    patch_manifest = root / "configs/gqf_official/patch_manifest.json"
    paper = root / (
        "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
        "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
    )
    official = root / "third_party/GQF"
    official_head: str | None = None
    official_tracked_tree_clean: bool | None = None
    head_path = official / ".git/HEAD"
    if head_path.exists():
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(official), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        official_head = completed.stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(official),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        official_tracked_tree_clean = not bool(status.stdout.strip())
    prereg_payload = _load_json(prereg)
    return {
        "paper": {
            "doi": PAPER_DOI,
            "arxiv": PAPER_ARXIV,
            "local_path": paper.relative_to(root).as_posix(),
            "local_sha256": _file_sha256(paper) if paper.exists() else None,
            "preregistered_sha256": prereg_payload["primary_sources"][
                "paper_and_supplement"
            ]["sha256"],
        },
        "official_gqf": {
            "url": "https://github.com/Matteo-Puviani/GQF",
            "required_commit": GQF_COMMIT,
            "live_commit": official_head,
            "live_commit_matches": official_head == GQF_COMMIT,
            "tracked_tree_clean": official_tracked_tree_clean,
            "license": "MIT",
        },
        "paper_exact_preregistration": {
            "path": prereg.relative_to(root).as_posix(),
            "sha256": _file_sha256(prereg),
        },
        "patch_manifest": {
            "path": patch_manifest.relative_to(root).as_posix(),
            "sha256": _file_sha256(patch_manifest),
        },
        "config_declared_source": config.get("source_registry", {}),
    }


def _runtime_profile() -> dict[str, Any]:
    signature, digest = _current_training_runtime()
    return {
        "schema_version": "t9.1.3-finalization-runtime-profile-v1",
        "role": "FINALIZATION_AND_LIVE_VALIDATION_RUNTIME",
        "signature": signature,
        "signature_sha256": digest,
        "rng_contract": (
            "explicit counter-derived simulator seeds; per-agent initialization seed; "
            "terminal torch CPU/CUDA and NumPy states retained"
        ),
    }


def _training_runtime_contract(
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]],
    runtime_profile: Mapping[str, Any],
    *,
    production: bool,
) -> dict[str, Any]:
    signatures: list[dict[str, Any]] = []
    hashes: list[str] = []
    for strategy in ("mf", "nmf"):
        for _, agent in agents[strategy]:
            signature = agent.get("training_runtime_signature")
            if not isinstance(signature, Mapping):
                raise CorruptAgentArtifactError(
                    f"{strategy} agent omitted its training runtime signature"
                )
            normalized, digest = _validated_training_runtime_signature(
                signature, label=f"{strategy} agent runtime contract"
            )
            if production:
                _validate_production_determinism_signature(normalized)
            if agent.get("training_runtime_signature_sha256") != digest:
                raise CorruptAgentArtifactError(
                    f"{strategy} agent runtime hash does not bind its signature"
                )
            signatures.append(normalized)
            hashes.append(digest)
    distinct = sorted(set(hashes))
    canonical_signature = signatures[0] if len(distinct) == 1 and signatures else None
    finalizer_hash = runtime_profile.get("signature_sha256")
    return {
        "schema_version": "t9.1.3-training-runtime-contract-v1",
        "signature": canonical_signature,
        "signature_sha256": distinct[0] if len(distinct) == 1 else None,
        "agent_signature_count": len(signatures),
        "distinct_agent_signature_sha256": distinct,
        "all_agents_identical": len(signatures) > 0 and len(distinct) == 1,
        "finalizer_signature_sha256": finalizer_hash,
        "equals_finalizer_runtime_signature": len(distinct) == 1
        and distinct[0] == finalizer_hash,
    }


def _gpu_load_attestation_report(
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]],
    finalizer_attestation: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    implementation_hash: str,
    runtime_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Close the training/finalizer launch proofs into one report evidence chain."""

    training_by_hash: dict[str, dict[str, Any]] = {}
    training_bindings: dict[str, dict[str, Any]] = {}
    for strategy in ("mf", "nmf"):
        for _, agent in agents[strategy]:
            attestation = _validate_gpu_load_attestation(
                agent.get("gpu_load_attestation"),
                config_sha256=_canonical_sha256(config),
                implementation_sha256=implementation_hash,
                expected_purpose="TRAINING_LAUNCH",
                current_runtime=runtime_signature,
                require_fresh=False,
                require_live_parent=False,
            )
            binding = _validated_gpu_attestation_binding(
                agent.get("gpu_load_attestation_binding")
            )
            expected_binding = _gpu_attestation_binding(attestation)
            if binding != expected_binding:
                raise CorruptAgentArtifactError(
                    f"{strategy} agent GPU attestation binding drifted"
                )
            digest = str(binding["attestation_sha256"])
            previous = training_by_hash.setdefault(digest, attestation)
            if previous != attestation:
                raise CorruptAgentArtifactError(
                    "one training GPU attestation hash maps to multiple payloads"
                )
            training_bindings[digest] = binding
    finalizer = _validate_gpu_load_attestation(
        finalizer_attestation,
        config_sha256=_canonical_sha256(config),
        implementation_sha256=implementation_hash,
        expected_purpose="FINALIZER_LAUNCH",
        current_runtime=runtime_signature,
        require_fresh=False,
        require_live_parent=False,
    )
    finalizer_binding = _gpu_attestation_binding(finalizer)
    ordered_hashes = sorted(training_by_hash)
    all_bindings = [training_bindings[digest] for digest in ordered_hashes] + [
        finalizer_binding
    ]
    target_uuids = {
        str(binding["target_gpu_uuid"]).lower() for binding in all_bindings
    }
    target_names = {str(binding["target_gpu_name"]) for binding in all_bindings}
    target_memories = {
        float(binding["target_gpu_total_memory_mib"]) for binding in all_bindings
    }
    return {
        "schema_version": "t9.1.3-gpu-load-attestation-report-v1",
        "training_launch_attestations": [
            training_by_hash[digest] for digest in ordered_hashes
        ],
        "training_launch_bindings": [
            training_bindings[digest] for digest in ordered_hashes
        ],
        "distinct_training_attestation_sha256": ordered_hashes,
        "finalizer_launch_attestation": finalizer,
        "finalizer_launch_binding": finalizer_binding,
        "all_forty_agents_bound": sum(len(agents[name]) for name in ("mf", "nmf"))
        == 40,
        "same_target_gpu_uuid_name_and_total_memory": (
            len(target_uuids) == len(target_names) == len(target_memories) == 1
        ),
        "freshness_scope": (
            "freshness and live supervisor-parent identity were enforced at each "
            "production process entry; report validation rechecks canonical/raw/device "
            "integrity without requiring historical attestations to remain unexpired"
        ),
    }


def _training_rows(
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in ("mf", "nmf"):
        for _, agent in agents[strategy]:
            record = agent["training_record"]
            for point in record["training_curve"]:
                rows.append(
                    {
                        "strategy": strategy,
                        "root_seed": int(agent["root_seed"]),
                        "epoch": int(point["epoch"]),
                        "mean_reward": float(point["mean_reward"]),
                        "score_path": float(point["score_path"]),
                        "loss": float(point["loss"]),
                        "gradient_norm_before_clip": float(
                            point["gradient_norm_before_clip"]
                        ),
                        "baseline": float(point["baseline"]),
                        "ground_outcome_fraction": float(
                            point["ground_outcome_fraction"]
                        ),
                        "control_residual_rms": float(point["control_residual_rms"]),
                        "control_slew_rms": float(point["control_slew_rms"]),
                        "training_trajectory_seed": int(
                            point["training_trajectory_seed"]
                        ),
                        "pre_model_sha256": point["pre_model_sha256"],
                        "post_model_sha256": point["post_model_sha256"],
                        "pre_optimizer_sha256": point["pre_optimizer_sha256"],
                        "post_optimizer_sha256": point["post_optimizer_sha256"],
                        "pre_rng_sha256": point["pre_rng_sha256"],
                        "post_rng_sha256": point["post_rng_sha256"],
                        "previous_lineage_sha256": point[
                            "previous_lineage_sha256"
                        ],
                        "epoch_metrics_sha256": point["epoch_metrics_sha256"],
                        "lineage_sha256": point["lineage_sha256"],
                        "status": "COMPLETE",
                    }
                )
    return rows


def _agent_registry_rows(
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]],
    selected: Mapping[str, tuple[Any, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in ("mf", "nmf"):
        selected_seed = int(selected[strategy][1]["root_seed"])
        for _, agent in agents[strategy]:
            training = agent["training_record"]
            gpu_binding = agent.get("gpu_load_attestation_binding") or {}
            rows.append(
                {
                    "strategy": strategy,
                    "root_seed": int(agent["root_seed"]),
                    "architecture": training["architecture"],
                    "parameter_count": int(training["parameter_count"]),
                    "epochs_executed": int(training["epochs_executed"]),
                    "training_trajectories": int(training["training_trajectories"]),
                    "best_validation_epoch": int(training["best_validation_epoch"]),
                    "best_validation_score": float(training["best_validation_score"]),
                    "selected_across_agents": int(agent["root_seed"]) == selected_seed,
                    "weight_path": agent["weight_path"],
                    "weight_file_sha256": agent["weight_file_sha256"],
                    "canonical_tensor_sha256": agent["canonical_tensor_sha256"],
                    "reload_witness": bool(agent["reload_witness"]),
                    "terminal_path": agent["terminal_path"],
                    "terminal_file_sha256": agent["terminal_file_sha256"],
                    "terminal_reload_witness": bool(
                        agent["terminal_reload_witness"]
                    ),
                    "validation_candidate_count": len(
                        training["validation_history"]
                    ),
                    "validation_candidate_reload_witness": bool(
                        agent["validation_candidate_reload_witness"]
                    ),
                    "training_lineage_reload_witness": bool(
                        agent["training_lineage_reload_witness"]
                    ),
                    "training_lineage_genesis_sha256": training[
                        "training_lineage_genesis_sha256"
                    ],
                    "training_lineage_terminal_sha256": training[
                        "training_lineage_terminal_sha256"
                    ],
                    "training_lineage_ledger_sha256": agent[
                        "live_training_lineage_validation"
                    ]["chain_ledger_sha256"],
                    "initial_state_sha256": training["initial_state_sha256"],
                    "initial_optimizer_sha256": training["training_curve"][0][
                        "pre_optimizer_sha256"
                    ],
                    "initial_rng_sha256": training["training_curve"][0][
                        "pre_rng_sha256"
                    ],
                    "terminal_state_sha256": training["terminal_state_sha256"],
                    "terminal_optimizer_sha256": training["training_curve"][-1][
                        "post_optimizer_sha256"
                    ],
                    "terminal_rng_sha256": training["training_curve"][-1][
                        "post_rng_sha256"
                    ],
                    "validation_candidate_physics_replay_witness": bool(
                        agent["live_validation_candidate_physics_replay"][
                            "all_candidates_replayed"
                        ]
                    ),
                    "validation_candidate_physics_replay_maximum_error": float(
                        agent["live_validation_candidate_physics_replay"][
                            "maximum_numeric_absolute_error"
                        ]
                    ),
                    "validation_candidate_physics_replay_ledger_sha256": agent[
                        "live_validation_candidate_physics_replay"
                    ]["candidate_replay_ledger_sha256"],
                    "training_runtime_signature_sha256": agent[
                        "training_runtime_signature_sha256"
                    ],
                    "training_gpu_attestation_sha256": gpu_binding.get(
                        "attestation_sha256"
                    ),
                    "training_gpu_attestation_purpose": gpu_binding.get(
                        "purpose"
                    ),
                    "training_gpu_attestation_transaction_id": gpu_binding.get(
                        "transaction_id"
                    ),
                    "training_gpu_attestation_target_uuid": gpu_binding.get(
                        "target_gpu_uuid"
                    ),
                    "training_gpu_attestation_target_name": gpu_binding.get(
                        "target_gpu_name"
                    ),
                    "training_gpu_attestation_target_total_memory_mib": gpu_binding.get(
                        "target_gpu_total_memory_mib"
                    ),
                    "training_gpu_attestation_sampling_completed_at_utc": gpu_binding.get(
                        "sampling_completed_at_utc"
                    ),
                    "training_gpu_attestation_expires_at_utc": gpu_binding.get(
                        "expires_at_utc"
                    ),
                    "failure_state": "NONE",
                }
            )
    return rows


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires a nonempty finite vector")
    return {
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "worst_quartile_mean": float(
            np.mean(np.sort(array)[: max(1, array.size // 4)])
        ),
    }


def _distribution_payload_valid(value: Mapping[str, Any], *, count: int) -> bool:
    try:
        values = value.get("values")
        return bool(
            isinstance(values, list)
            and len(values) == count
            and value == _distribution(values)
        )
    except (TypeError, ValueError):
        return False


def _artifact_manifest(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    entries = []
    for path in paths:
        resolved = path.resolve()
        entries.append(
            {
                "path": resolved.relative_to(root.resolve()).as_posix(),
                "sha256": _file_sha256(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    entries.sort(key=lambda item: item["path"])
    return {
        "files": entries,
        "sorted_manifest_sha256": _canonical_sha256(entries),
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
    }


def _training_failure_matches_start(
    failure: Mapping[str, Any],
    start: Mapping[str, Any] | None,
    attempt_directory: Path,
    *,
    config: Mapping[str, Any],
    implementation_hash: str,
) -> bool:
    """Bind a failed training terminal marker to its own immutable start marker."""

    try:
        if start is None:
            return False
        attempt_id = _attempt_id(attempt_directory)
        strategy = str(start["strategy"])
        seed = int(start["root_seed"])
        expected_start_path = _portable_artifact_path(
            attempt_directory / "start.json"
        )
        common = bool(
            start.get("schema_version")
            == "t9.1.3-training-attempt-start-v1"
            and start.get("task_id") == TASK_ID
            and start.get("attempt_id") == attempt_id
            and failure.get("attempt_id") == attempt_id
            and failure.get("training_attempt_start_path") == expected_start_path
            and failure.get("strategy") == strategy
            and int(failure.get("root_seed", -1)) == seed
            and failure.get("config_sha256") == start.get("config_sha256")
            == _canonical_sha256(config)
            and failure.get("implementation_sha256")
            == start.get("implementation_sha256")
            == implementation_hash
            and failure.get("parent_analysis_sha256")
            == start.get("parent_analysis_sha256")
            == T9_1_1_ANALYSIS_SHA256
            and _training_gpu_attestation_binding_valid(
                start.get("gpu_load_attestation_binding"), config
            )
        )
        if not common:
            return False
        if start.get("runtime_signature_state") == "CAPTURED_BEFORE_TRAINING":
            start_signature = start.get("training_runtime_signature")
            failure_signature = failure.get("training_runtime_signature")
            if not isinstance(start_signature, Mapping) or not isinstance(
                failure_signature, Mapping
            ):
                return False
            normalized, runtime_hash = _validated_training_runtime_signature(
                start_signature, label=f"attempt start {attempt_id}"
            )
            return bool(
                dict(failure_signature) == normalized
                and start.get("training_runtime_signature_sha256")
                == runtime_hash
                and failure.get("training_runtime_signature_sha256")
                == runtime_hash
                and start.get("agent_contract_hash")
                == failure.get("agent_contract_hash")
                == _agent_contract_hash(
                    config, strategy, seed, implementation_hash, normalized
                )
            )
        return bool(
            start.get("runtime_signature_state")
            == "UNKNOWN_START_COMMIT_INTERRUPTED"
            and start.get("state") == "RECOVERED_START_MARKER_INTERRUPTED"
            and start.get("training_runtime_signature") is None
            and start.get("training_runtime_signature_sha256") is None
            and start.get("agent_contract_hash") is None
            and failure.get("failure_subtype") == "START_MARKER_INTERRUPTED"
        )
    except (KeyError, TypeError, ValueError, AgentArtifactError):
        return False


def _failure_attempt_valid(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    implementation_hash: str,
) -> bool:
    try:
        strategy = str(row["strategy"])
        seed = int(row["root_seed"])
        path = _resolve_repo_path(str(row["path"]))
        stored_signature = row.get("training_runtime_signature")
        if not isinstance(stored_signature, Mapping):
            return False
        _, stored_runtime_hash = _validated_training_runtime_signature(
            stored_signature, label=f"failure attempt {strategy} seed {seed}"
        )
        allowed_seeds = {int(value) for value in config["training"]["paired_root_seeds"]}
        if (
            row.get("schema_version") != "t9.1.3-agent-failure-v1"
            or row.get("task_id") != TASK_ID
            or strategy not in {"mf", "nmf"}
            or seed not in allowed_seeds
            or row.get("failure_state")
            not in {"NULL", "TIMEOUT", "OOM", "NUMERIC_FAILURE", "CORRUPT_SHARD"}
            or not row.get("failure_subtype")
            or not row.get("exception_type")
            or row.get("retained_instead_of_silent_drop") is not True
            or row.get("safe_retry_authorized") is not True
            or row.get("config_sha256") != _canonical_sha256(config)
            or row.get("implementation_sha256") != implementation_hash
            or row.get("parent_analysis_sha256") != T9_1_1_ANALYSIS_SHA256
            or row.get("training_runtime_signature_sha256")
            != stored_runtime_hash
            or row.get("agent_contract_hash")
            != _agent_contract_hash(
                config, strategy, seed, implementation_hash, stored_signature
            )
            or not path.is_file()
            or row.get("file_sha256") != _file_sha256(path)
            or int(row.get("file_bytes", -1)) != path.stat().st_size
        ):
            return False
        archived = row.get("archived_partial_files", [])
        if not isinstance(archived, list):
            return False
        if row.get("failure_state") == "CORRUPT_SHARD" and not archived:
            return False
        for entry in archived:
            archived_path = _resolve_repo_path(str(entry["path"]))
            if (
                not archived_path.is_file()
                or int(entry["bytes"]) != archived_path.stat().st_size
                or entry["sha256"] != _file_sha256(archived_path)
            ):
                return False
        root = Path(__file__).resolve().parents[2]
        relative_parts = path.resolve().relative_to(root.resolve()).parts
        if "_training_attempts" in relative_parts:
            start_path = path.parent / "start.json"
            start = _load_json(start_path) if start_path.is_file() else None
            if not _training_failure_matches_start(
                row,
                start,
                path.parent,
                config=config,
                implementation_hash=implementation_hash,
            ):
                return False
    except (KeyError, TypeError, ValueError, OSError, FileNotFoundError):
        return False
    return True


def _audit_output_tree(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    implementation_hash: str,
) -> dict[str, Any]:
    """Classify every regular output byte; unknown/orphan files remain visible."""

    root = Path(__file__).resolve().parents[2]
    output_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and not _is_canonical_finalize_lock_management_file(path, output_dir)
    )
    expected_agent_paths = {
        path.resolve()
        for strategy in ("mf", "nmf")
        for seed in config["training"]["paired_root_seeds"]
        for path in _agent_paths(output_dir, strategy, int(seed))
    }
    failure_paths = set(path.resolve() for path in _failure_paths(output_dir))
    failure_attempts: list[dict[str, Any]] = []
    declared_archived_paths: set[Path] = set()
    for failure_path in sorted(failure_paths):
        failure = _load_json(failure_path)
        failure["path"] = failure_path.relative_to(root.resolve()).as_posix()
        failure["file_sha256"] = _file_sha256(failure_path)
        failure["file_bytes"] = failure_path.stat().st_size
        failure_attempts.append(failure)
        for entry in failure.get("archived_partial_files", []):
            declared_archived_paths.add(_resolve_repo_path(entry["path"]).resolve())

    classified = expected_agent_paths | failure_paths | declared_archived_paths
    lock_history = output_dir / "_lock_history"
    if lock_history.exists():
        for directory in sorted(
            path for path in lock_history.iterdir() if path.is_dir()
        ):
            owner_path = directory / "owner.json"
            recovery_path = directory / "recovery.json"
            try:
                try:
                    owner = _load_json(owner_path) if owner_path.is_file() else None
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    owner = None
                recovery = _load_json(recovery_path)
                archived_files = recovery.get("archived_files")
                declared = {
                    _resolve_repo_path(str(entry["path"])).resolve()
                    for entry in archived_files
                }
                actual = {
                    path.resolve()
                    for path in directory.rglob("*")
                    if path.is_file() and path != recovery_path
                }
                valid_history = bool(
                    recovery.get("schema_version")
                    == "t9.1.3-finalize-lock-recovery-v1"
                    and recovery.get("task_id") == TASK_ID
                    and recovery.get("stale_owner") == owner
                    and recovery.get("owner_sha256")
                    == (_file_sha256(owner_path) if owner_path.is_file() else None)
                    and (
                        owner is None
                        or (
                            owner.get("schema_version")
                            == "t9.1.3-finalize-lock-v1"
                            and owner.get("task_id") == TASK_ID
                        )
                    )
                    and isinstance(archived_files, list)
                    and declared == actual
                    and all(
                        int(entry["bytes"])
                        == _resolve_repo_path(str(entry["path"])).stat().st_size
                        and entry["sha256"]
                        == _file_sha256(_resolve_repo_path(str(entry["path"])))
                        for entry in archived_files
                    )
                )
            except (
                FileNotFoundError,
                KeyError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                OSError,
            ):
                valid_history = False
            if valid_history:
                classified.update(declared | {recovery_path.resolve()})
    finalization_recovery = output_dir / "_finalization_recovery"
    if finalization_recovery.exists():
        for directory in sorted(
            path for path in finalization_recovery.iterdir() if path.is_dir()
        ):
            recovery_path = directory / "recovery.json"
            try:
                recovery = _load_json(recovery_path)
                entries = recovery["archived_files"]
                declared = {
                    _resolve_repo_path(str(entry["path"])).resolve()
                    for entry in entries
                }
                actual = {
                    path.resolve()
                    for path in directory.rglob("*")
                    if path.is_file() and path != recovery_path
                }
                valid_recovery = bool(
                    recovery.get("schema_version")
                    == "t9.1.3-finalization-atomic-recovery-v1"
                    and recovery.get("task_id") == TASK_ID
                    and isinstance(recovery.get("original_paths"), list)
                    and declared == actual
                    and all(
                        int(entry["bytes"])
                        == _resolve_repo_path(str(entry["path"])).stat().st_size
                        and entry["sha256"]
                        == _file_sha256(_resolve_repo_path(str(entry["path"])))
                        for entry in entries
                    )
                )
            except (
                FileNotFoundError,
                KeyError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                OSError,
            ):
                valid_recovery = False
            if valid_recovery:
                classified.update(declared | {recovery_path.resolve()})
    training_attempts: list[dict[str, Any]] = []
    attempt_root = output_dir / "_training_attempts"
    if attempt_root.exists():
        for directory in sorted(path for path in attempt_root.iterdir() if path.is_dir()):
            strategy, seed = _parse_attempt_identity(directory)
            start_path = directory / "start.json"
            outcome_path = directory / "outcome.json"
            failure_path = directory / "failure.json"
            superseded_path = directory / "superseded.json"
            start = _load_json(start_path) if start_path.is_file() else None
            outcome = _load_json(outcome_path) if outcome_path.is_file() else None
            attempt_failure = (
                _load_json(failure_path) if failure_path.is_file() else None
            )
            superseded = (
                _load_json(superseded_path) if superseded_path.is_file() else None
            )
            terminal_count = int(outcome_path.is_file()) + int(failure_path.is_file())
            allowed_seed = seed in {
                int(value) for value in config["training"]["paired_root_seeds"]
            }
            start_runtime_state = (
                start.get("runtime_signature_state") if start else None
            )
            start_runtime_hash: str | None = None
            captured_runtime_valid = False
            if start_runtime_state == "CAPTURED_BEFORE_TRAINING" and isinstance(
                start.get("training_runtime_signature"), Mapping
            ):
                try:
                    stored_signature, start_runtime_hash = (
                        _validated_training_runtime_signature(
                            start["training_runtime_signature"],
                            label=f"output attempt {directory.name}",
                        )
                    )
                    captured_runtime_valid = bool(
                        start.get("training_runtime_signature_sha256")
                        == start_runtime_hash
                        and start.get("agent_contract_hash")
                        == _agent_contract_hash(
                            config,
                            strategy,
                            seed,
                            implementation_hash,
                            stored_signature,
                        )
                        and _training_gpu_attestation_binding_valid(
                            start.get("gpu_load_attestation_binding"), config
                        )
                    )
                except (AgentArtifactError, TypeError, ValueError):
                    captured_runtime_valid = False
            recovered_unknown_start = bool(
                start_runtime_state == "UNKNOWN_START_COMMIT_INTERRUPTED"
                and start.get("state") == "RECOVERED_START_MARKER_INTERRUPTED"
                and start.get("training_runtime_signature") is None
                and start.get("training_runtime_signature_sha256") is None
                and start.get("agent_contract_hash") is None
                and start.get("gpu_load_attestation_binding") is None
                and start.get("started_at_utc") is None
                and isinstance(start.get("recovered_at_utc"), str)
                and attempt_failure
                and attempt_failure.get("failure_subtype")
                == "START_MARKER_INTERRUPTED"
                and attempt_failure.get("attempt_id") == _attempt_id(directory)
            )
            common_start_valid = bool(
                start
                and allowed_seed
                and start.get("schema_version")
                == "t9.1.3-training-attempt-start-v1"
                and start.get("task_id") == TASK_ID
                and start.get("strategy") == strategy
                and int(start.get("root_seed", -1)) == seed
                and start.get("config_sha256") == _canonical_sha256(config)
                and start.get("implementation_sha256") == implementation_hash
                and start.get("parent_analysis_sha256") == T9_1_1_ANALYSIS_SHA256
                and start.get("attempt_id") == _attempt_id(directory)
            )
            valid_start = common_start_valid and (
                captured_runtime_valid or recovered_unknown_start
            )
            record_path = _agent_paths(output_dir, strategy, seed)[2]
            canonical_record = (
                _load_json(record_path) if outcome is not None and record_path.is_file() else None
            )
            basic_outcome = bool(
                outcome
                and outcome.get("schema_version")
                == "t9.1.3-training-attempt-outcome-v1"
                and outcome.get("task_id") == TASK_ID
                and outcome.get("strategy") == strategy
                and int(outcome.get("root_seed", -1)) == seed
                and outcome.get("attempt_id") == _attempt_id(directory)
                and outcome.get("state") == "SUCCESS"
                and captured_runtime_valid
                and outcome.get("agent_contract_hash")
                == start.get("agent_contract_hash")
                and outcome.get("training_runtime_signature_sha256")
                == start_runtime_hash
                and outcome.get("gpu_load_attestation_binding")
                == start.get("gpu_load_attestation_binding")
            )
            valid_active_outcome = bool(
                basic_outcome
                and superseded is None
                and canonical_record
                and canonical_record.get("attempt_id") == _attempt_id(directory)
                and canonical_record.get("training_runtime_signature_sha256")
                == start_runtime_hash
                and canonical_record.get("gpu_load_attestation_binding")
                == start.get("gpu_load_attestation_binding")
                and outcome.get("canonical_tensor_sha256")
                == canonical_record.get("canonical_tensor_sha256")
                and outcome.get("weight_file_sha256")
                == canonical_record.get("weight_file_sha256")
                and outcome.get("terminal_file_sha256")
                == canonical_record.get("terminal_file_sha256")
            )
            superseded_archive = (
                _resolve_repo_path(str(superseded.get("archive_failure_path")))
                if superseded is not None
                else None
            )
            valid_superseded_outcome = bool(
                basic_outcome
                and superseded
                and superseded.get("schema_version")
                == "t9.1.3-attempt-superseded-v1"
                and superseded.get("task_id") == TASK_ID
                and superseded.get("strategy") == strategy
                and int(superseded.get("root_seed", -1)) == seed
                and superseded.get("attempt_id") == _attempt_id(directory)
                and superseded.get("state") == "SUPERSEDED_CORRUPT"
                and superseded_archive is not None
                and superseded_archive.is_file()
                and superseded.get("archive_failure_sha256")
                == _file_sha256(superseded_archive)
                and superseded.get("training_runtime_signature_sha256")
                == start_runtime_hash
            )
            failure_for_validation = None
            if attempt_failure is not None and failure_path.is_file():
                failure_for_validation = dict(attempt_failure)
                failure_for_validation.update(
                    {
                        "path": failure_path.relative_to(root.resolve()).as_posix(),
                        "file_sha256": _file_sha256(failure_path),
                        "file_bytes": failure_path.stat().st_size,
                    }
                )
            valid_failure_outcome = bool(
                outcome is None
                and failure_for_validation is not None
                and _failure_attempt_valid(
                    failure_for_validation,
                    config=config,
                    implementation_hash=implementation_hash,
                )
                and _training_failure_matches_start(
                    attempt_failure,
                    start,
                    directory,
                    config=config,
                    implementation_hash=implementation_hash,
                )
            )
            valid_outcome = valid_failure_outcome or (
                outcome is not None
                and (valid_active_outcome or valid_superseded_outcome)
            )
            training_attempts.append(
                {
                    "path": directory.resolve().relative_to(root.resolve()).as_posix(),
                    "strategy": strategy,
                    "root_seed": seed,
                    "valid_start": valid_start,
                    "valid_outcome": valid_outcome,
                    "runtime_signature_state": start_runtime_state,
                    "training_runtime_signature_sha256": start_runtime_hash,
                    "terminal_marker_count": terminal_count,
                    "state": (
                        "SUPERSEDED_CORRUPT"
                        if superseded is not None
                        else "SUCCESS"
                        if outcome is not None
                        else "FAILURE"
                    ),
                }
            )
            if start_path.is_file():
                classified.add(start_path.resolve())
            if outcome_path.is_file():
                classified.add(outcome_path.resolve())
            if superseded_path.is_file():
                classified.add(superseded_path.resolve())

    unknown = [
        path.relative_to(root.resolve()).as_posix()
        for path in output_files
        if path.resolve() not in classified
    ]
    successful_cells = Counter(
        (row["strategy"], row["root_seed"])
        for row in training_attempts
        if row["state"] == "SUCCESS"
    )
    return {
        "output_files": output_files,
        "output_file_count": len(output_files),
        "failure_attempts": failure_attempts,
        "training_attempts": training_attempts,
        "unknown_or_orphan_files": unknown,
        "all_training_attempts_terminal_and_valid": all(
            row["valid_start"]
            and row["valid_outcome"]
            and row["terminal_marker_count"] == 1
            for row in training_attempts
        ),
        "every_agent_has_successful_attempt": all(
            successful_cells[(strategy, int(seed))] >= 1
            for strategy in ("mf", "nmf")
            for seed in config["training"]["paired_root_seeds"]
        ),
    }


def _empirical_reconstruction_valid(reconstruction: Mapping[str, Any]) -> bool:
    try:
        expected_strata = set(reconstruction)
        if len(expected_strata) != 2:
            return False
        for methods in reconstruction.values():
            if set(methods) != {"standard", "mf", "nmf"}:
                return False
            for diagnostic in methods.values():
                tomography = diagnostic[
                    "empirical_linear_inversion_subchannel_diagnostic"
                ]
                numeric_arrays = (
                    np.asarray(tomography["ptm"], dtype=np.float64),
                    np.asarray(tomography["choi_real"], dtype=np.float64),
                    np.asarray(tomography["choi_imag"], dtype=np.float64),
                    np.asarray(tomography["tni_effect_eigenvalues"], dtype=np.float64),
                )
                if (
                    numeric_arrays[0].shape != (4, 4)
                    or numeric_arrays[1].shape != (4, 4)
                    or numeric_arrays[2].shape != (4, 4)
                    or numeric_arrays[3].shape != (2,)
                    or not all(np.all(np.isfinite(value)) for value in numeric_arrays)
                ):
                    return False
                scalar_fields = (
                    "pair_sum_linearity_residual",
                    "minimum_choi_eigenvalue",
                    "maximum_output_hermiticity_error",
                    "minimum_output_eigenvalue",
                    "minimum_survival",
                    "maximum_survival",
                    "mean_leakage",
                    "survival_spread",
                )
                if not all(np.isfinite(float(tomography[name])) for name in scalar_fields):
                    return False
                if (
                    diagnostic.get("empirical_cptni_passed_is_not_a_hard_gate")
                    is not True
                    or diagnostic.get("paper_lifetime_value") is not None
                    or diagnostic.get("paper_lifetime_state") != PAPER_NUMERIC_STATE
                ):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _event_timeline_valid(
    event_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    half_cycles: int,
) -> bool:
    """Validate causal pending-action reuse and the terminal VR correction."""

    try:
        trajectories = {str(row["trajectory_id"]): row for row in trajectory_rows}
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in event_rows:
            if list(row) != list(EVENT_ROW_KEYS):
                return False
            grouped.setdefault(str(row["trajectory_id"]), []).append(row)
        if set(grouped) != set(trajectories):
            return False
        nominal = (
            nominal_sbs_parameters(device="cpu", dtype=_require_torch().float64)
            .detach()
            .cpu()
            .numpy()
        )
        residual_bounds = np.full((15,), 2.0, dtype=np.float64)
        residual_bounds[-1] = 1.0
        for trajectory_id, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: int(row["half_cycle"]))
            if [int(row["half_cycle"]) for row in ordered] != list(
                range(1, half_cycles + 1)
            ):
                return False
            trajectory = trajectories[trajectory_id]
            strategy = str(trajectory["strategy"])
            if strategy not in {"standard", "mf", "nmf"}:
                return False
            decoded: list[tuple[np.ndarray, np.ndarray]] = []
            for index, row in enumerate(ordered):
                half = index + 1
                terminal = half == half_cycles
                if not (
                    int(row["layer_control_decision_prefix_length"]) == index
                    and int(row["physical_layer_application_half_cycle"]) == half
                    and int(row["measurement_outcome_half_cycle"]) == half
                    and int(row["post_measurement_control_decision_prefix_length"])
                    == half
                    and int(row["post_measurement_virtual_rotation_after_half_cycle"])
                    == half
                    and row["post_measurement_control_layer_application_half_cycle"]
                    == (None if terminal else half + 1)
                    and row[
                        "uses_current_measurement_outcome_for_post_measurement_control"
                    ]
                    is True
                    and row["terminal_feedback_correction"] is terminal
                    and row["evaluation_cluster_id"]
                    == trajectory["evaluation_cluster_id"]
                    and row["rng_stream_id"] == trajectory["rng_stream_id"]
                    and np.isclose(
                        float(row["full_cycle_coordinate"]),
                        0.5 * half,
                        rtol=0.0,
                        atol=0.0,
                    )
                    and int(row["outcome_g0_e1"]) in {0, 1}
                    and np.isfinite(float(row["selected_branch_probability"]))
                    and 0.0 < float(row["selected_branch_probability"]) <= 1.0
                    and row["postselected"] is False
                    and row["included_in_denominator"] is True
                    and row["development_only"] is True
                    and row["status"] == "COMPLETE"
                ):
                    return False
                layer = np.asarray(
                    json.loads(str(row["project_action_order_json"])),
                    dtype=np.float64,
                )
                post = np.asarray(
                    json.loads(str(row["post_measurement_project_action_order_json"])),
                    dtype=np.float64,
                )
                layer_gqf = np.asarray(
                    json.loads(str(row["gqf_action_order_json"])),
                    dtype=np.float64,
                )
                post_gqf = np.asarray(
                    json.loads(str(row["post_measurement_gqf_action_order_json"])),
                    dtype=np.float64,
                )
                if (
                    layer.shape != (15,)
                    or post.shape != (15,)
                    or not np.all(np.isfinite(layer))
                    or not np.all(np.isfinite(post))
                    or not np.array_equal(layer_gqf, project_to_gqf_action(layer))
                    or not np.array_equal(post_gqf, project_to_gqf_action(post))
                    or row["project_action_sha256"]
                    != hashlib.sha256(np.ascontiguousarray(layer).tobytes()).hexdigest()
                    or row["post_measurement_project_action_sha256"]
                    != hashlib.sha256(np.ascontiguousarray(post).tobytes()).hexdigest()
                    or np.any(np.abs(layer - nominal) > residual_bounds + 1.0e-12)
                    or np.any(np.abs(post - nominal) > residual_bounds + 1.0e-12)
                    or (
                        strategy == "standard"
                        and (
                            not np.allclose(layer, nominal, rtol=0.0, atol=1.0e-14)
                            or not np.allclose(post, nominal, rtol=0.0, atol=1.0e-14)
                        )
                    )
                ):
                    return False
                decoded.append((layer, post))
            if any(
                not np.array_equal(decoded[index][1], decoded[index + 1][0])
                for index in range(half_cycles - 1)
            ):
                return False
            if sum(bool(row["terminal_feedback_correction"]) for row in ordered) != 1:
                return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _six_state_raw_replay_audit_valid(
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    trajectory_count: int,
    event_count: int,
) -> bool:
    """Validate the sealed seeded same-runtime/same-backend replay witness."""

    q = config["six_state_qualification"]
    expected_groups = 3 * len(STATE_LABELS) * (
        len(q["seeds"]) + len(q["confirmation_seeds"])
    )
    expected_scope = (
        "seeded same-runtime/same-backend replay of selected checkpoints, RNG, "
        "physics, actions and raw rows; not cross-runtime bitwise reproducibility "
        "and not an independent second physics backend"
    )
    expected_keys = {
        "schema_version",
        "scope",
        "group_count",
        "trajectory_count",
        "event_count",
        "maximum_action_absolute_error",
        "maximum_branch_probability_absolute_error",
        "maximum_projected_rho_absolute_error",
        "all_rows_replayed",
    }
    try:
        errors = tuple(
            float(audit[name])
            for name in (
                "maximum_action_absolute_error",
                "maximum_branch_probability_absolute_error",
                "maximum_projected_rho_absolute_error",
            )
        )
        return bool(
            set(audit) == expected_keys
            and audit["schema_version"]
            == "t9.1.3-six-state-same-backend-replay-v1"
            and audit["scope"] == expected_scope
            and int(audit["group_count"]) == expected_groups
            and int(audit["trajectory_count"]) == trajectory_count
            and int(audit["event_count"]) == event_count
            and audit["all_rows_replayed"] is True
            and all(np.isfinite(value) and value >= 0.0 for value in errors)
            and errors[0] <= 1.0e-11
            and errors[1] <= 1.0e-11
            and errors[2] <= 5.0e-10
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _training_lineage_gate_valid(
    training_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    expected_epochs: int,
) -> bool:
    """Independently re-close the exported lineage ledgers for the gate table."""

    try:
        registry = {
            (str(row["strategy"]), int(row["root_seed"])): row
            for row in agent_rows
        }
        if len(registry) != len(agent_rows):
            return False
        grouped_training: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
        for row in training_rows:
            grouped_training.setdefault(
                (str(row["strategy"]), int(row["root_seed"])), []
            ).append(row)
        grouped_selection: dict[
            tuple[str, int], list[Mapping[str, Any]]
        ] = {}
        for row in selection_rows:
            grouped_selection.setdefault(
                (str(row["strategy"]), int(row["root_seed"])), []
            ).append(row)
        if set(grouped_training) != set(registry) or set(grouped_selection) != set(
            registry
        ):
            return False
        for key, registry_row in registry.items():
            strategy, seed = key
            rows = sorted(grouped_training[key], key=lambda row: int(row["epoch"]))
            if (
                registry_row.get("training_lineage_reload_witness") is not True
                or len(rows) != expected_epochs
                or [int(row["epoch"]) for row in rows]
                != list(range(1, expected_epochs + 1))
            ):
                return False
            genesis = _training_lineage_genesis_sha256(
                strategy=strategy,
                seed=seed,
                initial_model_sha256=registry_row["initial_state_sha256"],
                initial_optimizer_sha256=registry_row[
                    "initial_optimizer_sha256"
                ],
                initial_rng_sha256=registry_row["initial_rng_sha256"],
            )
            if genesis != registry_row["training_lineage_genesis_sha256"]:
                return False
            previous_lineage = genesis
            previous_model = registry_row["initial_state_sha256"]
            previous_optimizer = registry_row["initial_optimizer_sha256"]
            previous_rng = registry_row["initial_rng_sha256"]
            chain_rows: list[dict[str, Any]] = []
            lineage_by_epoch = {0: genesis}
            model_by_epoch = {0: previous_model}
            for row in rows:
                epoch = int(row["epoch"])
                internal = {
                    **row,
                    "lineage_strategy": strategy,
                    "lineage_root_seed": seed,
                }
                if (
                    row.get("status") != "COMPLETE"
                    or int(row["training_trajectory_seed"])
                    != _training_trajectory_seed(seed, epoch)
                    or row["pre_model_sha256"] != previous_model
                    or row["pre_optimizer_sha256"] != previous_optimizer
                    or row["pre_rng_sha256"] != previous_rng
                    or row["previous_lineage_sha256"] != previous_lineage
                    or row["epoch_metrics_sha256"]
                    != _training_metric_sha256(row)
                    or row["lineage_sha256"]
                    != _training_epoch_lineage_sha256(internal)
                ):
                    return False
                previous_model = row["post_model_sha256"]
                previous_optimizer = row["post_optimizer_sha256"]
                previous_rng = row["post_rng_sha256"]
                previous_lineage = row["lineage_sha256"]
                lineage_by_epoch[epoch] = previous_lineage
                model_by_epoch[epoch] = previous_model
                chain_rows.append(
                    {
                        "epoch": epoch,
                        "lineage_sha256": previous_lineage,
                        "post_model_sha256": previous_model,
                        "post_optimizer_sha256": previous_optimizer,
                        "post_rng_sha256": previous_rng,
                    }
                )
            if (
                previous_model != registry_row["terminal_state_sha256"]
                or previous_optimizer
                != registry_row["terminal_optimizer_sha256"]
                or previous_rng != registry_row["terminal_rng_sha256"]
                or previous_lineage
                != registry_row["training_lineage_terminal_sha256"]
                or _canonical_sha256(chain_rows)
                != registry_row["training_lineage_ledger_sha256"]
            ):
                return False
            for selection in grouped_selection[key]:
                epoch = int(selection["checkpoint_epoch"])
                if (
                    epoch not in lineage_by_epoch
                    or selection.get("training_lineage_sha256")
                    != lineage_by_epoch[epoch]
                    or selection.get("checkpoint_tensor_sha256")
                    != model_by_epoch[epoch]
                ):
                    return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _six_state_raw_replay_audits_compatible(
    config: Mapping[str, Any],
    sealed: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    trajectory_count: int,
    event_count: int,
) -> bool:
    """Compare replay witnesses without pretending maxima are cross-process bitwise.

    Categorical fields and denominators remain exact.  Each maximum must first
    satisfy its preregistered acceptance threshold; the sealed/live maxima are
    then compared with a much tighter tolerance than that acceptance threshold.
    """

    if not (
        _six_state_raw_replay_audit_valid(
            config,
            sealed,
            trajectory_count=trajectory_count,
            event_count=event_count,
        )
        and _six_state_raw_replay_audit_valid(
            config,
            live,
            trajectory_count=trajectory_count,
            event_count=event_count,
        )
    ):
        return False
    categorical = (
        "schema_version",
        "scope",
        "group_count",
        "trajectory_count",
        "event_count",
        "all_rows_replayed",
    )
    if any(sealed.get(name) != live.get(name) for name in categorical):
        return False
    tolerances = {
        "maximum_action_absolute_error": 1.0e-14,
        "maximum_branch_probability_absolute_error": 1.0e-14,
        "maximum_projected_rho_absolute_error": 5.0e-13,
    }
    try:
        return all(
            np.isclose(
                float(sealed[name]),
                float(live[name]),
                rtol=1.0e-6,
                atol=absolute_tolerance,
            )
            for name, absolute_tolerance in tolerances.items()
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _build_gates(
    *,
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    sources: Mapping[str, Any],
    deviations: Sequence[Mapping[str, Any]],
    agents: Mapping[str, Sequence[tuple[Any, Mapping[str, Any]]]],
    agent_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    sentinel_retrain_audit: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    raw_replay_audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    failure_attempts: Sequence[Mapping[str, Any]],
    implementation_hash: str,
    training_runtime_contract: Mapping[str, Any],
    output_audit: Mapping[str, Any],
    expected_manifest_file_count: int,
) -> dict[str, bool]:
    th = _require_torch()
    roots = tuple(int(seed) for seed in config["training"]["paired_root_seeds"])
    q = config["six_state_qualification"]
    derived_training = {
        _training_trajectory_seed(seed, epoch)
        for seed in roots
        for epoch in range(0, int(config["training"]["epochs"]) + 1)
    }
    configured_nontraining = (
        set(int(seed) for seed in config["training"]["validation_seeds"])
        | set(int(seed) for seed in q["seeds"])
        | set(int(seed) for seed in q["confirmation_seeds"])
    )
    expected_trajectories = 3 * len(STATE_LABELS) * (
        len(q["seeds"]) * int(q["batch_size"])
        + len(q["confirmation_seeds"]) * int(q["confirmation_batch_size"])
    )
    expected_events = expected_trajectories * 2 * int(q["full_cycles"])
    expected_validation_points = 1 + int(config["training"]["epochs"]) // int(
        config["training"]["validation_interval"]
    )
    expected_validation_epochs = {0} | set(
        range(
            int(config["training"]["validation_interval"]),
            int(config["training"]["epochs"]) + 1,
            int(config["training"]["validation_interval"]),
        )
    )
    expected_selection_keys = {
        (strategy, seed, epoch)
        for strategy in ("mf", "nmf")
        for seed in roots
        for epoch in expected_validation_epochs
    }
    actual_selection_keys = [
        (
            str(row["strategy"]),
            int(row["root_seed"]),
            int(row["checkpoint_epoch"]),
        )
        for row in selection_rows
    ]
    serious = [row for row in deviations if row.get("severity") in {"High", "Critical"}]
    scores = {
        strategy: [
            float(item[1]["training_record"]["best_validation_score"])
            for item in agents[strategy]
        ]
        for strategy in ("mf", "nmf")
    }
    expected_training_keys = {
        (strategy, seed, epoch)
        for strategy in ("mf", "nmf")
        for seed in roots
        for epoch in range(1, int(config["training"]["epochs"]) + 1)
    }
    actual_training_keys = [
        (str(row["strategy"]), int(row["root_seed"]), int(row["epoch"]))
        for row in training_rows
    ]
    primary_name = f"primary_cutoff{int(config['training']['cutoff'])}"
    confirmation_name = (
        f"selected_confirmation_cutoff{int(config['training']['confirmation_cutoff'])}"
    )
    stratum_specs = {
        primary_name: (
            int(config["training"]["cutoff"]),
            tuple(int(seed) for seed in q["seeds"]),
            int(q["batch_size"]),
        ),
        confirmation_name: (
            int(config["training"]["confirmation_cutoff"]),
            tuple(int(seed) for seed in q["confirmation_seeds"]),
            int(q["confirmation_batch_size"]),
        ),
    }
    expected_trajectory_keys = {
        (stratum, strategy, state, seed, member)
        for stratum, (_, seeds, batch_size) in stratum_specs.items()
        for strategy in ("standard", "mf", "nmf")
        for state in STATE_LABELS
        for seed in seeds
        for member in range(batch_size)
    }
    actual_trajectory_keys = [
        (
            str(row["stratum"]),
            str(row["strategy"]),
            str(row["initial_state"]),
            int(row["evaluation_seed"]),
            int(row["batch_member"]),
        )
        for row in trajectory_rows
    ]
    trajectory_ids = [str(row["trajectory_id"]) for row in trajectory_rows]
    trajectory_id_set = set(trajectory_ids)
    event_counts = Counter(str(row["trajectory_id"]) for row in event_rows)
    event_half_cycles: dict[str, set[int]] = {}
    for row in event_rows:
        event_half_cycles.setdefault(str(row["trajectory_id"]), set()).add(
            int(row["half_cycle"])
        )
    reconstruction_entries = [
        diagnostic
        for methods in reconstruction.values()
        for diagnostic in methods.values()
    ]
    gates = {
        "identity_and_schema_frozen": config.get("task_id") == TASK_ID,
        "parent_protocol_hash_live": parent.get("analysis_sha256")
        == T9_1_1_ANALYSIS_SHA256,
        "paper_constrained_namespace_only": config.get("evidence_grade")
        == EVIDENCE_GRADE,
        "official_commit_pinned": sources["official_gqf"]["required_commit"]
        == GQF_COMMIT,
        "official_live_commit_matches_when_present": sources["official_gqf"][
            "live_commit"
        ]
        in {None, GQF_COMMIT}
        and sources["official_gqf"]["tracked_tree_clean"] in {None, True},
        "paper_bytes_match_preregistered_source": sources["paper"][
            "local_sha256"
        ]
        == sources["paper"]["preregistered_sha256"],
        "deviation_ledger_is_nontrivial": (
            len(deviations) == 42
            and [row["id"] for row in deviations]
            == [f"D{index:02d}" for index in range(1, 43)]
        ),
        "serious_deviations_revoke_numeric_anchor": len(serious) == 39
        and all(row.get("paper_numeric_anchor_eligible") is False for row in serious),
        "twenty_unique_paired_roots": len(roots) == len(set(roots)) == 20,
        "derived_training_and_evaluation_seed_domains_disjoint": not (
            derived_training & configured_nontraining
        ),
        "both_families_have_twenty_agents": all(
            len(agents[strategy]) == 20 for strategy in ("mf", "nmf")
        ),
        "all_forty_training_units_accounted": len(agent_rows) == 40,
        "all_agent_training_runtime_signatures_identical_and_bound": (
            training_runtime_contract.get("agent_signature_count") == 40
            and training_runtime_contract.get("all_agents_identical") is True
            and training_runtime_contract.get("equals_finalizer_runtime_signature")
            is True
            and isinstance(training_runtime_contract.get("signature_sha256"), str)
            and len(str(training_runtime_contract["signature_sha256"])) == 64
            and training_runtime_contract.get("distinct_agent_signature_sha256")
            == [training_runtime_contract.get("signature_sha256")]
            and isinstance(training_runtime_contract.get("signature"), Mapping)
            and all(
                row.get("training_runtime_signature_sha256")
                == training_runtime_contract["signature_sha256"]
                for row in agent_rows
            )
        ),
        "paper_scale_architectures": all(
            (row["strategy"] == "mf" and row["architecture"] == PAPER_MF_ARCHITECTURE)
            or (row["strategy"] == "nmf" and row["architecture"] == PAPER_RNN_ARCHITECTURE)
            for row in agent_rows
        ),
        "parameter_counts_match": all(
            (row["strategy"] == "mf" and row["parameter_count"] == 70_159)
            or (row["strategy"] == "nmf" and row["parameter_count"] == 72_853)
            for row in agent_rows
        ),
        "all_agents_executed_1000_epochs": all(
            row["epochs_executed"] == 1000 for row in agent_rows
        ),
        "all_weights_hash_and_reload": all(
            row["reload_witness"]
            and row["terminal_reload_witness"]
            and row["validation_candidate_reload_witness"]
            and row["validation_candidate_count"] == 26
            and len(row["weight_file_sha256"]) == 64
            and len(row["terminal_file_sha256"]) == 64
            and len(row["canonical_tensor_sha256"]) == 64
            for row in agent_rows
        ),
        "all_gradients_and_training_metrics_finite": len(training_rows) == 40_000
        and all(
            np.isfinite(row["gradient_norm_before_clip"])
            and row["gradient_norm_before_clip"] > 0.0
            and np.isfinite(row["loss"])
            for row in training_rows
        ),
        "per_agent_epoch_census_exact": (
            len(actual_training_keys) == len(set(actual_training_keys))
            and set(actual_training_keys) == expected_training_keys
        ),
        "all_training_epochs_lineage_closed_and_endpoints_bound": (
            _training_lineage_gate_valid(
                training_rows,
                agent_rows,
                selection_rows,
                expected_epochs=int(config["training"]["epochs"]),
            )
        ),
        "frozen_family_sentinel_full_retrain_matches_epoch_lineage_and_terminal": (
            _sentinel_retrain_audit_valid(
                sentinel_retrain_audit,
                expected_epochs=int(config["training"]["epochs"]),
                expected_population_agents=40,
                expected_roots={"mf": roots[0], "nmf": roots[0]},
            )
        ),
        "validation_candidate_census_complete": len(selection_rows)
        == 40 * expected_validation_points
        and len(actual_selection_keys) == len(set(actual_selection_keys))
        and set(actual_selection_keys) == expected_selection_keys,
        "all_validation_candidates_replayed_from_checkpoint_and_frozen_seeds": all(
            row["validation_candidate_physics_replay_witness"] is True
            and np.isfinite(
                row["validation_candidate_physics_replay_maximum_error"]
            )
            and row["validation_candidate_physics_replay_maximum_error"]
            <= 2.0e-10
            and _is_lower_sha256(
                row["validation_candidate_physics_replay_ledger_sha256"]
            )
            for row in agent_rows
        ),
        "checkpoint_selection_validation_only": all(
            row["selection_split"] == "validation_only"
            and row["test_or_formal_used"] is False
            for row in selection_rows
        ),
        "one_selected_snapshot_per_agent_and_family": all(
            sum(
                row["selected_within_agent"]
                for row in selection_rows
                if row["strategy"] == strategy
                and row["root_seed"] == seed
            )
            == 1
            for strategy in ("mf", "nmf")
            for seed in roots
        )
        and all(
            sum(
                row["selected_across_agents"]
                for row in selection_rows
                if row["strategy"] == strategy
            )
            == 1
            for strategy in ("mf", "nmf")
        )
        and all(
            len(row["checkpoint_tensor_sha256"]) == 64 for row in selection_rows
        ),
        "one_selected_agent_per_family": all(
            sum(
                row["selected_across_agents"]
                for row in agent_rows
                if row["strategy"] == strategy
            )
            == 1
            for strategy in ("mf", "nmf")
        ),
        "deterministic_family_tie_break": all(
            len(scores[strategy]) == 20 and np.all(np.isfinite(scores[strategy]))
            for strategy in ("mf", "nmf")
        ),
        "six_states_exact_and_balanced": (
            set(row["initial_state"] for row in trajectory_rows)
            == set(STATE_LABELS)
            and len(actual_trajectory_keys) == len(set(actual_trajectory_keys))
            and set(actual_trajectory_keys) == expected_trajectory_keys
        ),
        "six_state_semantics_and_paired_clusters_disclosed": all(
            row["state_definition_id"] == q["state_definition_id"]
            and row["logical_operator_definition_id"]
            == q["logical_operator_definition_id"]
            and row["development_only"] is True
            and row["evaluation_cluster_id"]
            == f"{row['stratum']}:seed_{int(row['evaluation_seed'])}"
            and row["rng_stream_id"]
            == (
                f"torch_cpu_seed_{int(row['evaluation_seed'])}:"
                f"member_{int(row['batch_member'])}"
            )
            and np.isfinite(row["raw_codeword_overlap_real"])
            and np.isfinite(row["raw_codeword_overlap_imag"])
            and np.isfinite(row["raw_logical_gram_condition_number"])
            and row["raw_logical_gram_condition_number"] >= 1.0
            for row in trajectory_rows
        )
        and q["monte_carlo_design"].startswith("paired_common_random_numbers"),
        "standard_mf_nmf_all_qualified": set(
            row["strategy"] for row in trajectory_rows
        )
        == {"standard", "mf", "nmf"},
        "trajectory_census_complete": len(trajectory_rows) == expected_trajectories,
        "event_census_complete": (
            len(event_rows) == expected_events
            and len(trajectory_ids) == len(trajectory_id_set)
            and set(event_counts) == trajectory_id_set
            and all(
                event_counts[trajectory_id] == 2 * int(q["full_cycles"])
                and event_half_cycles[trajectory_id]
                == set(range(1, 2 * int(q["full_cycles"]) + 1))
                for trajectory_id in trajectory_id_set
            )
            and all(int(row["outcome_g0_e1"]) in {0, 1} for row in event_rows)
        ),
        "raw_schema_matches_frozen_trajectory_and_event_contracts": (
            list(trajectory_rows[0]) == list(TRAJECTORY_ROW_KEYS)
            and list(event_rows[0]) == list(EVENT_ROW_KEYS)
            and all(list(row) == list(TRAJECTORY_ROW_KEYS) for row in trajectory_rows)
            and all(list(row) == list(EVENT_ROW_KEYS) for row in event_rows)
        ),
        "no_postselection_full_denominator": all(
            row["postselected"] is False and row["included_in_denominator"] is True
            for row in trajectory_rows
        )
        and all(
            row["postselected"] is False and row["included_in_denominator"] is True
            for row in event_rows
        ),
        "all_trajectory_statuses_retained": all(
            row["status"] == "COMPLETE" for row in trajectory_rows
        ),
        "branch_probabilities_valid": all(
            np.isfinite(row["selected_branch_probability"])
            and 0.0 < row["selected_branch_probability"] <= 1.0
            for row in event_rows
        ),
        "event_action_timing_is_causal_and_explicit": (
            config["training"]["feedback_timeline"]
            == PAPER_CAUSAL_FEEDBACK_TIMELINE
            and _event_timeline_valid(
                event_rows,
                trajectory_rows,
                half_cycles=2 * int(q["full_cycles"]),
            )
        ),
        "six_state_raw_rows_replay_selected_checkpoints_rng_and_physics": (
            _six_state_raw_replay_audit_valid(
                config,
                raw_replay_audit,
                trajectory_count=expected_trajectories,
                event_count=expected_events,
            )
        ),
        "code_survival_and_fidelity_finite": all(
            np.isfinite(row["code_survival"])
            and -1.0e-8 <= row["code_survival"] <= 1.0 + 1.0e-8
            and abs(row["code_weighted_I"] - row["code_survival"]) <= 1.0e-10
            and np.isfinite(row["state_fidelity_unconditional"])
            and -1.0e-9
            <= row["state_fidelity_unconditional"]
            <= row["code_survival"] + 1.0e-9
            and all(
                abs(row[f"code_weighted_{axis}"])
                <= row["code_survival"] + 1.0e-9
                for axis in ("X", "Y", "Z")
            )
            for row in trajectory_rows
        ),
        "six_state_fidelity_pauli_identity": all(
            abs(
                2.0 * row["state_fidelity_unconditional"]
                - row["code_survival"]
                - row["signed_axis_signal"]
            )
            <= 1.0e-9
            for row in trajectory_rows
        ),
        "normalized_conditional_joint_state_invariants": bool(reconstruction_entries)
        and all(
            np.isfinite(item["maximum_joint_trace_error"])
            and item["maximum_joint_trace_error"] <= 2.0e-9
            and np.isfinite(item["maximum_joint_hermiticity_error"])
            and item["maximum_joint_hermiticity_error"] <= 2.0e-9
            and np.isfinite(
                item["maximum_projected_hermiticity_error_before_stabilization"]
            )
            and item["maximum_projected_hermiticity_error_before_stabilization"]
            <= 2.0e-9
            and np.isfinite(item["minimum_final_joint_eigenvalue"])
            and item["minimum_final_joint_eigenvalue"] >= -2.0e-8
            for item in reconstruction_entries
        ),
        "empirical_linear_inversion_diagnostics_finite_nonpromotional": (
            _empirical_reconstruction_valid(reconstruction)
        ),
        "action_permutation_roundtrip": np.array_equal(
            gqf_to_project_action(project_to_gqf_action(np.arange(15))),
            np.arange(15),
        ),
        "action_order_nominal_and_bounds_match_live_backend": (
            tuple(config["controller_contract"]["action_order"])
            == PROJECT_ACTION_ORDER
            and np.allclose(
                np.asarray(config["controller_contract"]["nominal_action"]),
                nominal_sbs_parameters(device="cpu", dtype=th.float64)
                .detach()
                .cpu()
                .numpy(),
                rtol=0.0,
                atol=1.0e-14,
            )
            and config["controller_contract"]["residual_bounds"][
                "indices_0_through_13"
            ]
            == [-2.0, 2.0]
            and config["controller_contract"]["residual_bounds"][
                "virtual_rotation_index_14"
            ]
            == [-1.0, 1.0]
        ),
        "all_published_files_manifested_exactly_once": (
            manifest.get("file_count", 0) == expected_manifest_file_count
            and len(
                {row["path"] for row in manifest.get("files", [])}
            )
            == manifest.get("file_count", 0)
        ),
        "output_tree_has_no_unknown_or_orphan_bytes": not output_audit.get(
            "unknown_or_orphan_files"
        ),
        "training_attempt_ledger_is_terminal_and_complete": bool(
            output_audit.get("training_attempts")
        )
        and output_audit.get("all_training_attempts_terminal_and_valid") is True
        and output_audit.get("every_agent_has_successful_attempt") is True,
        "all_failure_attempts_are_retained_and_typed": all(
            _failure_attempt_valid(
                row,
                config=config,
                implementation_hash=implementation_hash,
            )
            for row in failure_attempts
        ),
        "paper_scale_numeric_is_null": config["paper_scale_result"]["state"]
        == PAPER_NUMERIC_STATE
        and config["paper_scale_result"]["value"] is None,
        "official_exact_is_null": config["external_claim_slots"][
            "official_exact"
        ]["value"]
        is None,
        "puviani_surpass_is_null": config["external_claim_slots"]["surpass"][
            "value"
        ]
        is None,
        "native_free_action_lane_not_ranked_phase9": config["ranking_eligibility"]
        == "INELIGIBLE_SIGNATURE_NONRANKING_HISTORICAL_BASELINE",
    }
    return gates


def _mutation_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Separate published-gate rejection from actual source-evidence mutations."""

    baseline = copy.deepcopy(dict(payload))
    baseline.setdefault(
        "mutation_audit",
        {"mutations": [], "mutation_count": 0, "detected_count": 0, "all_detected": False},
    )
    baseline.pop("analysis_sha256", None)
    baseline["analysis_sha256"] = _canonical_sha256(baseline)
    validate_report(
        baseline,
        verify_live_files=False,
        _require_mutation_audit=False,
        _structural_test_only=True,
    )
    rows: list[dict[str, Any]] = []
    for gate in payload["required_gates"]:
        mutated = copy.deepcopy(dict(payload))
        mutated.setdefault(
            "mutation_audit",
            {"mutations": [], "mutation_count": 0, "detected_count": 0, "all_detected": False},
        )
        mutated["gates"][gate] = False
        mutated.pop("analysis_sha256", None)
        mutated["analysis_sha256"] = _canonical_sha256(mutated)
        detected = False
        reason: str | None = None
        try:
            validate_report(
                mutated,
                verify_live_files=False,
                _require_mutation_audit=False,
                _structural_test_only=True,
            )
        except (ValueError, TypeError, FileNotFoundError) as error:
            reason = str(error)
            detected = "analysis hash mismatch" not in reason
        rows.append(
            {
                "mutation_id": f"published_gate_false:{gate}",
                "target": "published_gate_boolean",
                "detected": detected,
                "detection_reason": reason,
            }
        )
    source_mutations = {
        "source_runtime_signature_tamper": lambda value: value[
            "training_runtime_contract"
        ]["signature"].update({"torch_version": "forged-runtime"}),
        "source_raw_replay_witness_tamper": lambda value: value[
            "six_state_evaluator"
        ]["raw_replay_audit"].update(
            {"maximum_projected_rho_absolute_error": 1.0e-3}
        ),
        "source_output_tree_orphan_injection": lambda value: value[
            "output_tree_audit"
        ]["unknown_or_orphan_files"].append("docs/t9_1_3_puviani_agents/orphan.bin"),
        "source_training_epoch_census_tamper": lambda value: value[
            "training_ledger"
        ].update({"rows": int(value["training_ledger"]["rows"]) - 1}),
        "source_six_state_denominator_tamper": lambda value: value[
            "six_state_evaluator"
        ].update(
            {
                "trajectory_rows": int(
                    value["six_state_evaluator"]["trajectory_rows"]
                )
                - 1
            }
        ),
        "source_failure_runtime_binding_tamper": lambda value: value[
            "failure_attempt_ledger"
        ].update({"all_failure_runtime_signatures_self_bound": False}),
        "source_deviation_ledger_hash_tamper": lambda value: value[
            "deviation_ledger"
        ].update({"sha256": "0" * 64}),
    }
    if tuple(source_mutations) != SOURCE_EVIDENCE_MUTATION_IDS:
        raise RuntimeError("source-evidence mutation registry drifted")
    for mutation_id, mutate in source_mutations.items():
        mutated = copy.deepcopy(dict(payload))
        mutated.setdefault(
            "mutation_audit",
            {
                "mutations": [],
                "mutation_count": 0,
                "detected_count": 0,
                "all_detected": False,
            },
        )
        mutate(mutated)
        mutated.pop("analysis_sha256", None)
        mutated["analysis_sha256"] = _canonical_sha256(mutated)
        detected = False
        reason = None
        try:
            validate_report(
                mutated,
                verify_live_files=False,
                _require_mutation_audit=False,
                _structural_test_only=True,
            )
        except (ValueError, TypeError, FileNotFoundError) as error:
            reason = str(error)
            detected = "analysis hash mismatch" not in reason
        rows.append(
            {
                "mutation_id": mutation_id,
                "target": "source_evidence",
                "detected": detected,
                "detection_reason": reason,
            }
        )
    claim_mutations = {
        "evidence_grade_escalation": lambda value: value.__setitem__(
            "evidence_grade", "OFFICIAL_EXACT_REPRODUCTION"
        ),
        "paper_numeric_injection": lambda value: value[
            "paper_scale_numerical_result"
        ].update({"state": "COMPLETE", "value": 1500.0}),
        "puviani_surpass_injection": lambda value: value["external_claim_slots"].update(
            {"puviani_surpass": True}
        ),
        "nested_paper_target_injection": lambda value: value["config"][
            "paper_scale_target"
        ]["numeric_values"].update({"T_X": 999.0}),
        "nested_external_claim_injection": lambda value: value["config"][
            "external_claim_slots"
        ]["surpass"].update({"value": {"gain": 1.1}}),
        "selection_split_injection": lambda value: value["config"]["training"][
            "selection"
        ].update({"selection_data": "test"}),
        "reconstruction_paper_lifetime_injection": lambda value: next(
            iter(
                next(
                    iter(value["six_state_evaluator"]["reconstruction"].values())
                ).values()
            )
        ).update({"paper_lifetime_value": 1500.0}),
        "namespace_official_exact_injection": lambda value: value["namespace"][
            "official_exact"
        ].update({"state": "COMPLETE", "value": 1500.0}),
        "published_anchor_eligibility_injection": lambda value: value[
            "paper_scale_numerical_result"
        ].update({"published_anchor_comparison_eligible": True}),
        "claim_boundary_deletion": lambda value: value["claim_boundary"].update(
            {"forbidden": ""}
        ),
        "formal_selection_injection": lambda value: value["selection_contract"].update(
            {"formal_used_for_selection": True}
        ),
        "production_contract_downgrade": lambda value: value.__setitem__(
            "production_contract", False
        ),
        "manifest_census_injection": lambda value: value["artifact_manifest"].update(
            {"file_count": int(value["artifact_manifest"]["file_count"]) + 1}
        ),
        "unknown_top_level_claim_injection": lambda value: value.update(
            {"claimed_paper_T_ch_cycles": 1500.0}
        ),
        "unknown_nested_evaluator_claim_injection": lambda value: value[
            "six_state_evaluator"
        ].update({"paper_lifetime_sota_cycles": 1500.0}),
        "action_ranking_eligibility_injection": lambda value: value[
            "action_contract"
        ].update({"ranking_eligibility": "ELIGIBLE_SOTA"}),
        "protocol_transcription_injection": lambda value: value[
            "protocol_transcription"
        ].update({"selection": "paper_test_best_of_20"}),
        "qualification_scope_injection": lambda value: value[
            "six_state_evaluator"
        ].update({"qualification_scope": "paper-scale lifetime SOTA"}),
        "selection_description_injection": lambda value: value[
            "selection_contract"
        ].update({"within_agent": "best test checkpoint"}),
    }
    if tuple(claim_mutations) != CLAIM_MUTATION_IDS:
        raise RuntimeError("claim mutation registry drifted")
    for mutation_id, mutate in claim_mutations.items():
        mutated = copy.deepcopy(dict(payload))
        mutated.setdefault(
            "mutation_audit",
            {"mutations": [], "mutation_count": 0, "detected_count": 0, "all_detected": False},
        )
        mutate(mutated)
        mutated.pop("analysis_sha256", None)
        mutated["analysis_sha256"] = _canonical_sha256(mutated)
        detected = False
        reason = None
        try:
            validate_report(
                mutated,
                verify_live_files=False,
                _require_mutation_audit=False,
                _structural_test_only=True,
            )
        except (ValueError, TypeError, FileNotFoundError) as error:
            reason = str(error)
            detected = "analysis hash mismatch" not in reason
        rows.append(
            {
                "mutation_id": mutation_id,
                "target": "claim_boundary",
                "detected": detected,
                "detection_reason": reason,
            }
        )
    return {
        "mutations": rows,
        "mutation_count": len(rows),
        "detected_count": sum(row["detected"] for row in rows),
        "all_detected": all(row["detected"] for row in rows),
    }


def _finalization_marker(
    config: Mapping[str, Any],
    *,
    status: str,
    started_at_utc: str,
    error: BaseException | None = None,
    forensic_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deliberately non-PASS marker for an in-flight/failed seal."""

    marker: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": status,
        "evidence_grade": EVIDENCE_GRADE,
        "started_at_utc": started_at_utc,
        "config_sha256": _canonical_sha256(config),
        "valid_pass_seal": False,
    }
    if error is not None:
        marker["failure_type"] = type(error).__name__
        marker["failure_message"] = str(error)
    if forensic_path is not None and forensic_path.is_file():
        marker["forensic_candidate"] = {
            "path": _portable_artifact_path(forensic_path),
            "sha256": _file_sha256(forensic_path),
            "bytes": forensic_path.stat().st_size,
        }
    return marker


def _forensic_candidate_path(report_target: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return report_target.with_name(
        f".{report_target.name}.forensic_candidate_{stamp}_{time.time_ns()}.json"
    )


def _write_invalid_forensic_candidate(
    candidate: Mapping[str, Any] | None,
    *,
    report_target: Path,
    error: BaseException,
) -> Path:
    """Retain the rejected candidate without leaving any candidate status PASS."""

    forensic_path = _forensic_candidate_path(report_target)
    forensic = copy.deepcopy(dict(candidate)) if candidate is not None else {}
    prior_analysis = forensic.pop("analysis_sha256", None)
    forensic["status"] = "INVALID_UNPUBLISHED_CANDIDATE_LIVE_VALIDATION_FAILED"
    forensic["valid_pass_seal"] = False
    forensic["forensic_validation_failure"] = {
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "candidate_analysis_sha256_before_invalidation": prior_analysis,
        "invalidated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(forensic, forensic_path)
    return forensic_path


def finalize_artifacts(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    report_path: str | Path,
    agent_registry_path: str | Path,
    selection_ledger_path: str | Path,
    training_ledger_path: str | Path,
    trajectory_path: str | Path,
    event_path: str | Path,
    production: bool = True,
    gpu_attestation: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate a candidate under one exclusive lock, then atomically publish PASS."""

    _reject_canonical_production_config_downgrade(config, production=production)
    if production:
        _configure_production_determinism()
    output = Path(output_dir)
    report_target = Path(report_path)
    with _namespace_finalize_lock(output):
        runtime_signature, _ = _validate_runtime_for_config(
            config, production=production
        )
        live_implementation_hash = implementation_sha256()
        normalized_gpu_attestation: dict[str, Any] | None = None
        if production:
            normalized_gpu_attestation = _validate_gpu_load_attestation(
                gpu_attestation,
                config_sha256=_canonical_sha256(config),
                implementation_sha256=live_implementation_hash,
                expected_purpose="FINALIZER_LAUNCH",
                current_runtime=runtime_signature,
                require_fresh=True,
                require_live_parent=True,
            )
        # This is intentionally the first namespace mutation after lock
        # acquisition.  Expensive recovery/replay may not leave an old PASS
        # visible while a new evidence generation is in progress.
        started_at_utc = datetime.now(timezone.utc).isoformat()
        _atomic_json(
            _finalization_marker(
                config,
                status="INVALIDATED_BEFORE_FINALIZATION_NO_VALID_SEAL",
                started_at_utc=started_at_utc,
            ),
            report_target,
        )
        candidate: dict[str, Any] | None = None
        try:
            _recover_incomplete_lock_history(output)
            _recover_incomplete_finalization_recoveries(output)
            candidate = _finalize_artifacts_locked(
                config,
                output_dir=output,
                report_path=report_target,
                agent_registry_path=agent_registry_path,
                selection_ledger_path=selection_ledger_path,
                training_ledger_path=training_ledger_path,
                trajectory_path=trajectory_path,
                event_path=event_path,
                production=production,
                gpu_attestation=normalized_gpu_attestation,
            )
            if candidate.get("status") == STATUS_PASS:
                # Do not call the public wrapper here: this namespace lock is
                # already held and recursive acquisition must fail closed.
                _validate_report_locked(candidate, verify_live_files=True)
            elif production:
                raise RuntimeError(
                    "production candidate did not satisfy the PASS gate contract"
                )
        except Exception as error:
            forensic_path = _write_invalid_forensic_candidate(
                candidate,
                report_target=report_target,
                error=error,
            )
            _atomic_json(
                _finalization_marker(
                    config,
                    status="FINALIZATION_FAILED_NO_VALID_SEAL",
                    started_at_utc=started_at_utc,
                    error=error,
                    forensic_path=forensic_path,
                ),
                report_target,
            )
            raise
        # The only write capable of publishing STATUS_PASS occurs after the
        # complete live-file validation above and remains inside this lock.
        _atomic_json(candidate, report_target)
        return candidate


def _finalize_artifacts_locked(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    report_path: str | Path,
    agent_registry_path: str | Path,
    selection_ledger_path: str | Path,
    training_ledger_path: str | Path,
    trajectory_path: str | Path,
    event_path: str | Path,
    production: bool = True,
    gpu_attestation: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Load all frozen agents, run six-state qualification, and seal the lane."""

    _reject_canonical_production_config_downgrade(config, production=production)
    if production:
        _configure_production_determinism()
    _validate_config(config, production=production)
    runtime_signature, _ = _validate_runtime_for_config(
        config, production=production
    )
    parent = _verify_parent_protocol(config)
    started = time.perf_counter()
    output = Path(output_dir)
    report_target = Path(report_path)
    agent_registry = Path(agent_registry_path)
    selection_ledger = Path(selection_ledger_path)
    training_ledger = Path(training_ledger_path)
    trajectories = Path(trajectory_path)
    events = Path(event_path)
    if production:
        for path, relative in (
            (output, "docs/t9_1_3_puviani_agents"),
            (report_target, "docs/t9_1_3_puviani_paper_constrained.json"),
            (agent_registry, "docs/t9_1_3_puviani_agent_registry.csv"),
            (selection_ledger, "docs/t9_1_3_puviani_selection_ledger.csv"),
            (training_ledger, "docs/t9_1_3_puviani_training_ledger.parquet"),
            (trajectories, "docs/t9_1_3_puviani_six_state_trajectories.parquet"),
            (events, "docs/t9_1_3_puviani_six_state_events.parquet"),
        ):
            _require_canonical_production_path(path, relative)
    live_implementation_hash = implementation_sha256()
    normalized_gpu_attestation: dict[str, Any] | None = None
    if production:
        normalized_gpu_attestation = _validate_gpu_load_attestation(
            gpu_attestation,
            config_sha256=_canonical_sha256(config),
            implementation_sha256=live_implementation_hash,
            expected_purpose="FINALIZER_LAUNCH",
            current_runtime=runtime_signature,
            require_fresh=True,
            require_live_parent=True,
        )
    _recover_finalization_atomic_temporaries(
        output,
        (
            report_target,
            agent_registry,
            selection_ledger,
            training_ledger,
            trajectories,
            events,
        ),
    )
    for strategy in ("mf", "nmf"):
        directional = _strategy_config(config, strategy)
        for seed in config["training"]["paired_root_seeds"]:
            _quarantine_agent_atomic_temporaries(
                output_dir=output,
                strategy=strategy,
                seed=int(seed),
                config=config,
                implementation_hash=live_implementation_hash,
            )
            _recover_incomplete_failed_attempts(
                output,
                config,
                live_implementation_hash,
                strategy_filter=strategy,
                seed_filter=int(seed),
            )
            _recover_incomplete_archives(
                output,
                config,
                live_implementation_hash,
                strategy_filter=strategy,
                seed_filter=int(seed),
            )
            _recover_interrupted_training_attempts(
                output_dir=output,
                strategy=strategy,
                seed=int(seed),
                config=config,
                directional=directional,
                implementation_hash=live_implementation_hash,
            )
    agents, implementation_hash = _all_agent_records(config, output)
    if implementation_hash != live_implementation_hash:
        raise RuntimeError("implementation hash changed during finalization")
    sentinel_retrain_audit = _sentinel_full_retrain_audit(
        config,
        agents,
        scratch_root=output / "_sentinel_retrain_scratch",
    )
    selected, selection_rows = _select_agents(agents)
    agent_rows = _agent_registry_rows(agents, selected)
    training_rows = _training_rows(agents)
    trajectory_rows, event_rows, reconstruction = _evaluate_six_state(config, selected)
    raw_replay_audit = _six_state_raw_replay_audit(
        config,
        selected,
        trajectory_rows,
        event_rows,
    )

    _write_csv(agent_rows, agent_registry)
    _write_csv(selection_rows, selection_ledger)
    _write_parquet(training_rows, training_ledger)
    _write_parquet(trajectory_rows, trajectories)
    _write_parquet(event_rows, events)

    deviation_path, deviation_payload, deviations = _validated_deviation_ledger(
        config
    )
    source_registry = _source_registry(config)
    runtime_profile = _runtime_profile()
    training_runtime_contract = _training_runtime_contract(
        agents, runtime_profile, production=production
    )
    gpu_load_attestations = (
        _gpu_load_attestation_report(
            agents,
            normalized_gpu_attestation,
            config=config,
            implementation_hash=implementation_hash,
            runtime_signature=runtime_profile["signature"],
        )
        if production and normalized_gpu_attestation is not None
        else None
    )
    root = Path(__file__).resolve().parents[2]
    shared_published_paths = [
        agent_registry,
        selection_ledger,
        training_ledger,
        trajectories,
        events,
        deviation_path,
        root / "configs/phase9/t9_1_3_puviani_paper_constrained.json",
        root / "docs/t9_1_1_three_lane_protocol.json",
        root / "docs/t9_1_1_three_lane_protocol_source_data.csv",
        root / "docs/phase9_three_lane_protocol.md",
        root
        / (
            "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
            "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
        ),
        *(root / relative for relative in IMPLEMENTATION_RELATIVE_PATHS),
        root / "configs/gqf_official/paper_exact_preregistration.json",
        root / "configs/gqf_official/patch_manifest.json",
    ]
    output_audit = _audit_output_tree(
        output_dir=output,
        config=config,
        implementation_hash=implementation_hash,
    )
    failure_attempts = output_audit["failure_attempts"]
    published_paths = shared_published_paths + list(output_audit["output_files"])
    manifest = _artifact_manifest(published_paths, root)
    expected_manifest_file_count = len(shared_published_paths) + int(
        output_audit["output_file_count"]
    )
    gates = _build_gates(
        config=config,
        parent=parent,
        sources=source_registry,
        deviations=deviations,
        agents=agents,
        agent_rows=agent_rows,
        selection_rows=selection_rows,
        trajectory_rows=trajectory_rows,
        event_rows=event_rows,
        training_rows=training_rows,
        sentinel_retrain_audit=sentinel_retrain_audit,
        reconstruction=reconstruction,
        raw_replay_audit=raw_replay_audit,
        manifest=manifest,
        failure_attempts=failure_attempts,
        implementation_hash=implementation_hash,
        training_runtime_contract=training_runtime_contract,
        output_audit=output_audit,
        expected_manifest_file_count=expected_manifest_file_count,
    )
    if tuple(gates) != REQUIRED_GATE_NAMES:
        raise RuntimeError("T9.1.3 gate registry drifted from the immutable order")
    required_gates = list(REQUIRED_GATE_NAMES)
    status = (
        STATUS_PASS
        if production and all(gates.values())
        else "FAIL_ARTIFACT_INTEGRITY"
        if production
        else "PILOT_ARTIFACT_NOT_A_PRODUCTION_SEAL"
    )
    selection_summary = _selection_summary(agents, selected)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "production_contract": bool(production),
        "evidence_grade": EVIDENCE_GRADE,
        "namespace": {
            "official_exact": {"state": "MISSING_EXTERNAL_ASSET", "value": None},
            "paper_constrained": "THIS_ARTIFACT",
            "project_native_directional": "T2.3.7_SEPARATE_NONTRANSFERABLE",
        },
        "t9_1_1_binding": parent,
        "implementation_sha256": implementation_hash,
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "source_registry": source_registry,
        "runtime_profile": runtime_profile,
        "training_runtime_contract": training_runtime_contract,
        "gpu_load_attestations": gpu_load_attestations,
        "protocol_transcription": copy.deepcopy(config["protocol_transcription"]),
        "action_contract": {
            "project_order": list(PROJECT_ACTION_ORDER),
            "gqf_order": list(GQF_ACTION_ORDER),
            "permutation_roundtrip": True,
            "bounds": "14 residuals in [-2,2], virtual rotation in [-1,1] after tanh",
            "ranking_eligibility": config["ranking_eligibility"],
        },
        "population": {
            "paired_root_count": len(config["training"]["paired_root_seeds"]),
            "total_agent_count": len(agent_rows),
            "agents_per_family": {"mf": len(agents["mf"]), "nmf": len(agents["nmf"])},
            "all_agents_published_not_only_winner": True,
            "agent_registry": _portable_artifact_path(agent_registry),
        },
        "selection_contract": {
            "within_agent": "max validation score; tie earlier epoch",
            "across_agents": "max best-validation score; tie earlier epoch then lower root seed",
            "formal_used_for_selection": False,
            "test_used_for_selection": False,
            "paper_best_of_20_test_selection_replaced": True,
            "selection_lock_sha256": _canonical_sha256(selection_rows),
            "ledger_path": _portable_artifact_path(selection_ledger),
            "selected": selection_summary,
        },
        "six_state_evaluator": {
            "state_labels": list(STATE_LABELS),
            "methods": ["standard", "mf", "nmf"],
            "trajectory_rows": len(trajectory_rows),
            "event_rows": len(event_rows),
            "trajectory_path": _portable_artifact_path(trajectories),
            "event_path": _portable_artifact_path(events),
            "postselection": False,
            "all_trajectories_retained": True,
            "development_only": True,
            "state_definition_id": config["six_state_qualification"][
                "state_definition_id"
            ],
            "logical_operator_definition_id": config[
                "six_state_qualification"
            ]["logical_operator_definition_id"],
            "monte_carlo_design": config["six_state_qualification"][
                "monte_carlo_design"
            ],
            "raw_replay_audit": raw_replay_audit,
            "reconstruction": reconstruction,
            "qualification_scope": (
                "finite-cutoff ten-cycle executable/raw qualification only; not a "
                "paper lifetime and not Phase9 >=1e4-cycle formal evidence"
            ),
        },
        "training_ledger": {
            "path": _portable_artifact_path(training_ledger),
            "rows": len(training_rows),
            "all_epochs_all_agents": True,
            "family_sentinel_full_retrain": sentinel_retrain_audit,
        },
        "failure_attempt_ledger": {
            "count": len(failure_attempts),
            "attempts": failure_attempts,
            "distinct_runtime_signature_sha256": sorted(
                {
                    str(row["training_runtime_signature_sha256"])
                    for row in failure_attempts
                }
            ),
            "all_failure_runtime_signatures_self_bound": all(
                _failure_attempt_valid(
                    row,
                    config=config,
                    implementation_hash=implementation_hash,
                )
                for row in failure_attempts
            ),
            "silent_drop": False,
            "note": (
                "A retained failed attempt does not replace the required completed agent; "
                "all 40 completed checkpoints remain mandatory."
            ),
        },
        "output_tree_audit": {
            "root": output.resolve().relative_to(root.resolve()).as_posix(),
            "output_file_count": output_audit["output_file_count"],
            "unknown_or_orphan_files": output_audit["unknown_or_orphan_files"],
            "training_attempt_count": len(output_audit["training_attempts"]),
            "training_attempts": output_audit["training_attempts"],
            "all_training_attempts_terminal_and_valid": output_audit[
                "all_training_attempts_terminal_and_valid"
            ],
            "every_agent_has_successful_attempt": output_audit[
                "every_agent_has_successful_attempt"
            ],
        },
        "deviation_ledger": {
            "path": deviation_path.relative_to(root).as_posix(),
            "sha256": _file_sha256(deviation_path),
            "entry_count": len(deviations),
            "critical_or_high_count": sum(
                row.get("severity") in {"High", "Critical"} for row in deviations
            ),
        },
        "paper_scale_numerical_result": {
            "state": PAPER_NUMERIC_STATE,
            "value": None,
            "reason": PAPER_NUMERIC_REASON,
            "required_cutoff": config["paper_scale_target"]["fock_cutoff"],
            "required_cycles": config["paper_scale_target"]["evaluation_full_cycles"],
            "current_qualification_cutoffs": [
                config["training"]["cutoff"],
                config["training"]["confirmation_cutoff"],
            ],
            "current_qualification_cycles": config["six_state_qualification"][
                "full_cycles"
            ],
            "published_anchor_comparison_eligible": False,
        },
        "external_claim_slots": {
            "official_exact": None,
            "puviani_surpass": None,
            "paper_T_X_T_Y_T_Z_T_ch": None,
        },
        "artifact_manifest": manifest,
        "gates": gates,
        "required_gates": required_gates,
        "claim_boundary": dict(REPORT_CLAIM_BOUNDARY),
        "wall_time_seconds_finalize": time.perf_counter() - started,
    }
    report["mutation_audit"] = _mutation_audit(report)
    if not report["mutation_audit"]["all_detected"]:
        raise RuntimeError("T9.1.3 semantic mutation audit failed")
    report["analysis_sha256"] = _canonical_sha256(report)
    return report


def _validated_manifest_paths(payload: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    manifest = payload.get("artifact_manifest")
    if not isinstance(manifest, dict) or set(manifest) != {
        "files",
        "sorted_manifest_sha256",
        "file_count",
        "total_bytes",
    }:
        raise TypeError("artifact manifest schema is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifact manifest must be nonempty")
    paths: list[str] = []
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise TypeError("artifact manifest entry schema is invalid")
        path = entry["path"]
        digest = entry["sha256"]
        size = entry["bytes"]
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or Path(path).is_absolute()
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("artifact manifest path is not canonical repo-relative")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("artifact manifest SHA-256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("artifact manifest byte count is invalid")
        paths.append(path)
        total_bytes += size
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("artifact manifest paths must be sorted and unique")
    if manifest.get("file_count") != len(entries):
        raise ValueError("artifact manifest file count mismatch")
    if manifest.get("total_bytes") != total_bytes:
        raise ValueError("artifact manifest total byte count mismatch")
    if manifest.get("sorted_manifest_sha256") != _canonical_sha256(entries):
        raise ValueError("artifact manifest root mismatch")
    path_set = set(paths)
    if not set(MANDATORY_SHARED_ARTIFACT_PATHS) <= path_set:
        raise ValueError("artifact manifest omits a mandatory shared artifact")
    config = payload["config"]
    expected_agent_paths = {
        f"docs/t9_1_3_puviani_agents/{strategy}/{strategy}_seed_{int(seed)}{suffix}"
        for strategy in ("mf", "nmf")
        for seed in config["training"]["paired_root_seeds"]
        for suffix in (".npz", ".terminal.pt", ".json")
    }
    if not expected_agent_paths <= path_set:
        raise ValueError("artifact manifest omits one or more canonical agent shards")
    return manifest, path_set


def _assert_exact_keys(
    value: Any, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise ValueError(
            f"{label} schema drifted: missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )
    return value


def _validate_report_exact_schema(
    payload: Mapping[str, Any], *, allow_missing_gpu_attestation: bool = False
) -> None:
    """Reject every unknown report field, including newly invented claims."""

    expected_top_level = set(REPORT_TOP_LEVEL_KEYS)
    if allow_missing_gpu_attestation and "gpu_load_attestations" not in payload:
        expected_top_level.remove("gpu_load_attestations")
    _assert_exact_keys(payload, expected_top_level, "report")
    _assert_exact_keys(payload["namespace"], {"official_exact", "paper_constrained", "project_native_directional"}, "namespace")
    _assert_exact_keys(payload["namespace"]["official_exact"], {"state", "value"}, "namespace.official_exact")

    sources = _assert_exact_keys(
        payload["source_registry"],
        {"paper", "official_gqf", "paper_exact_preregistration", "patch_manifest", "config_declared_source"},
        "source_registry",
    )
    _assert_exact_keys(sources["paper"], {"doi", "arxiv", "local_path", "local_sha256", "preregistered_sha256"}, "source_registry.paper")
    _assert_exact_keys(sources["official_gqf"], {"url", "required_commit", "live_commit", "live_commit_matches", "tracked_tree_clean", "license"}, "source_registry.official_gqf")
    _assert_exact_keys(sources["paper_exact_preregistration"], {"path", "sha256"}, "source_registry.paper_exact_preregistration")
    _assert_exact_keys(sources["patch_manifest"], {"path", "sha256"}, "source_registry.patch_manifest")

    _assert_exact_keys(
        payload["runtime_profile"],
        {
            "schema_version", "role", "signature", "signature_sha256",
            "rng_contract",
        },
        "runtime_profile",
    )
    runtime_profile = payload["runtime_profile"]
    _assert_exact_keys(
        runtime_profile["signature"],
        TRAINING_RUNTIME_SIGNATURE_KEYS,
        "runtime_profile.signature",
    )
    training_runtime = _assert_exact_keys(
        payload["training_runtime_contract"],
        {
            "schema_version", "signature", "signature_sha256",
            "agent_signature_count", "distinct_agent_signature_sha256",
            "all_agents_identical", "finalizer_signature_sha256",
            "equals_finalizer_runtime_signature",
        },
        "training_runtime_contract",
    )
    _assert_exact_keys(
        training_runtime["signature"],
        TRAINING_RUNTIME_SIGNATURE_KEYS,
        "training_runtime_contract.signature",
    )
    if "gpu_load_attestations" in payload:
        gpu_report = _assert_exact_keys(
            payload["gpu_load_attestations"],
            {
                "schema_version",
                "training_launch_attestations",
                "training_launch_bindings",
                "distinct_training_attestation_sha256",
                "finalizer_launch_attestation",
                "finalizer_launch_binding",
                "all_forty_agents_bound",
                "same_target_gpu_uuid_name_and_total_memory",
                "freshness_scope",
            },
            "gpu_load_attestations",
        )
        if not isinstance(gpu_report["training_launch_attestations"], list) or not isinstance(
            gpu_report["training_launch_bindings"], list
        ):
            raise TypeError("GPU training attestation report rows must be lists")
        for index, value in enumerate(gpu_report["training_launch_attestations"]):
            _assert_exact_keys(
                value,
                GPU_LOAD_ATTESTATION_KEYS,
                f"gpu_load_attestations.training[{index}]",
            )
        for index, value in enumerate(gpu_report["training_launch_bindings"]):
            _assert_exact_keys(
                value,
                GPU_LOAD_ATTESTATION_BINDING_KEYS,
                f"gpu_load_attestations.training_binding[{index}]",
            )
        _assert_exact_keys(
            gpu_report["finalizer_launch_attestation"],
            GPU_LOAD_ATTESTATION_KEYS,
            "gpu_load_attestations.finalizer",
        )
        _assert_exact_keys(
            gpu_report["finalizer_launch_binding"],
            GPU_LOAD_ATTESTATION_BINDING_KEYS,
            "gpu_load_attestations.finalizer_binding",
        )
    action = _assert_exact_keys(
        payload["action_contract"],
        {"project_order", "gqf_order", "permutation_roundtrip", "bounds", "ranking_eligibility"},
        "action_contract",
    )
    population = _assert_exact_keys(
        payload["population"],
        {"paired_root_count", "total_agent_count", "agents_per_family", "all_agents_published_not_only_winner", "agent_registry"},
        "population",
    )
    _assert_exact_keys(population["agents_per_family"], {"mf", "nmf"}, "population.agents_per_family")

    selection = _assert_exact_keys(
        payload["selection_contract"],
        {
            "within_agent", "across_agents", "formal_used_for_selection",
            "test_used_for_selection", "paper_best_of_20_test_selection_replaced",
            "selection_lock_sha256", "ledger_path", "selected",
        },
        "selection_contract",
    )
    selected = _assert_exact_keys(selection["selected"], {"mf", "nmf"}, "selection_contract.selected")
    distribution_keys = {"values", "mean", "median", "q1", "q3", "minimum", "maximum", "worst_quartile_mean"}
    for strategy in ("mf", "nmf"):
        item = _assert_exact_keys(
            selected[strategy],
            {
                "selected_root_seed", "selected_checkpoint_epoch",
                "selected_validation_score", "checkpoint_sha256", "agent_count",
                "validation_candidates_per_agent", "total_agent_checkpoint_candidates",
                "all_agent_best_validation_score_distribution",
                "all_agent_initial_validation_score_distribution",
                "all_agent_initial_to_best_gain_distribution",
                "all_agent_selected_checkpoint_epoch_distribution",
                "selection_optimism_warning",
            },
            f"selection_contract.selected.{strategy}",
        )
        for distribution_name in (
            "all_agent_best_validation_score_distribution",
            "all_agent_initial_validation_score_distribution",
            "all_agent_initial_to_best_gain_distribution",
            "all_agent_selected_checkpoint_epoch_distribution",
        ):
            _assert_exact_keys(
                item[distribution_name],
                distribution_keys,
                f"selection_contract.selected.{strategy}.{distribution_name}",
            )

    evaluator = _assert_exact_keys(
        payload["six_state_evaluator"],
        {
            "state_labels", "methods", "trajectory_rows", "event_rows",
            "trajectory_path", "event_path", "postselection",
            "all_trajectories_retained", "development_only", "state_definition_id",
            "logical_operator_definition_id", "monte_carlo_design", "reconstruction",
            "raw_replay_audit", "qualification_scope",
        },
        "six_state_evaluator",
    )
    _assert_exact_keys(
        evaluator["raw_replay_audit"],
        {
            "schema_version", "scope", "group_count", "trajectory_count",
            "event_count", "maximum_action_absolute_error",
            "maximum_branch_probability_absolute_error",
            "maximum_projected_rho_absolute_error", "all_rows_replayed",
        },
        "six_state_evaluator.raw_replay_audit",
    )
    config = payload["config"]
    expected_strata = {
        f"primary_cutoff{int(config['training']['cutoff'])}",
        f"selected_confirmation_cutoff{int(config['training']['confirmation_cutoff'])}",
    }
    reconstruction = _assert_exact_keys(evaluator["reconstruction"], expected_strata, "six_state_evaluator.reconstruction")
    diagnostic_keys = {
        "selected_agent_seed", "trajectory_count",
        "empirical_linear_inversion_subchannel_diagnostic",
        "monte_carlo_pair_sum_not_deterministic_tolerance",
        "empirical_cptni_passed_is_not_a_hard_gate", "state_definition_id",
        "logical_operator_definition_id", "monte_carlo_design",
        "raw_codeword_overlap_real", "raw_codeword_overlap_imag",
        "raw_logical_gram_condition_number", "diagnostic_scope",
        "maximum_joint_trace_error", "maximum_joint_hermiticity_error",
        "maximum_projected_hermiticity_error_before_stabilization",
        "minimum_final_joint_eigenvalue", "paper_lifetime_value",
        "paper_lifetime_state",
    }
    tomography_keys = {
        "ptm", "choi_real", "choi_imag", "tni_effect_eigenvalues",
        "pair_sum_linearity_residual", "minimum_choi_eigenvalue",
        "maximum_output_hermiticity_error", "minimum_output_eigenvalue",
        "minimum_survival", "maximum_survival", "mean_leakage", "survival_spread",
    }
    for stratum, methods in reconstruction.items():
        method_map = _assert_exact_keys(methods, {"standard", "mf", "nmf"}, f"reconstruction.{stratum}")
        for method, diagnostic in method_map.items():
            row = _assert_exact_keys(diagnostic, diagnostic_keys, f"reconstruction.{stratum}.{method}")
            _assert_exact_keys(row["empirical_linear_inversion_subchannel_diagnostic"], tomography_keys, f"reconstruction.{stratum}.{method}.tomography")

    _assert_exact_keys(
        payload["training_ledger"],
        {
            "path",
            "rows",
            "all_epochs_all_agents",
            "family_sentinel_full_retrain",
        },
        "training_ledger",
    )
    failure_ledger = _assert_exact_keys(payload["failure_attempt_ledger"], {"count", "attempts", "distinct_runtime_signature_sha256", "all_failure_runtime_signatures_self_bound", "silent_drop", "note"}, "failure_attempt_ledger")
    if not isinstance(failure_ledger["attempts"], list):
        raise TypeError("failure_attempt_ledger.attempts must be a list")
    failure_required = {
        "schema_version", "task_id", "strategy", "root_seed", "failure_state",
        "failure_subtype", "exception_type", "exception_message", "traceback",
        "config_sha256", "agent_contract_hash", "implementation_sha256",
        "training_runtime_signature", "training_runtime_signature_sha256",
        "parent_analysis_sha256", "archived_partial_files",
        "retained_instead_of_silent_drop", "safe_retry_authorized",
        "generated_at_utc", "path", "file_sha256", "file_bytes",
    }
    failure_optional = {
        "archive_completed_at_utc", "archive_recovered_after_interruption",
        "stale_lock_metadata", "orphan_atomic_sources",
        "training_attempt_start_path", "attempt_id",
    }
    for index, attempt in enumerate(failure_ledger["attempts"]):
        if not isinstance(attempt, Mapping):
            raise TypeError(f"failure attempt {index} must be a mapping")
        keys = set(attempt)
        if not failure_required <= keys or keys - failure_required - failure_optional:
            raise ValueError(f"failure attempt {index} schema drifted")
        if not isinstance(attempt["archived_partial_files"], list):
            raise TypeError("archived_partial_files must be a list")
        for entry in attempt["archived_partial_files"]:
            _assert_exact_keys(entry, {"path", "sha256", "bytes"}, "archived_partial_file")
        if "stale_lock_metadata" in attempt:
            stale = attempt["stale_lock_metadata"]
            stale_keys = set(stale) if isinstance(stale, Mapping) else set()
            current_lock_keys = {
                "schema_version", "task_id", "strategy", "root_seed", "pid",
                "process_created_unix_ns", "hostname", "token",
                "agent_contract_hash", "training_runtime_signature_sha256",
                "started_at_utc", "last_heartbeat_unix_ns",
                "last_heartbeat_utc", "heartbeat_sequence", "lease_seconds",
            }
            legacy_lock_keys = current_lock_keys - {
                "process_created_unix_ns", "last_heartbeat_unix_ns",
                "last_heartbeat_utc", "heartbeat_sequence", "lease_seconds",
            }
            pre_runtime_lock_keys = legacy_lock_keys - {
                "training_runtime_signature_sha256"
            }
            parse_error_keys = {"pid", "hostname", "parse_error"}
            if frozenset(stale_keys) not in {
                frozenset(current_lock_keys),
                frozenset(legacy_lock_keys),
                frozenset(pre_runtime_lock_keys),
                frozenset(parse_error_keys),
            }:
                raise ValueError("stale writer-lock metadata schema drifted")
        if "orphan_atomic_sources" in attempt and not (
            isinstance(attempt["orphan_atomic_sources"], list)
            and all(isinstance(value, str) for value in attempt["orphan_atomic_sources"])
        ):
            raise TypeError("orphan atomic source ledger is invalid")

    output_audit = _assert_exact_keys(
        payload["output_tree_audit"],
        {"root", "output_file_count", "unknown_or_orphan_files", "training_attempt_count", "training_attempts", "all_training_attempts_terminal_and_valid", "every_agent_has_successful_attempt"},
        "output_tree_audit",
    )
    if not isinstance(output_audit["training_attempts"], list):
        raise TypeError("output_tree_audit.training_attempts must be a list")
    for attempt in output_audit["training_attempts"]:
        _assert_exact_keys(attempt, {"path", "strategy", "root_seed", "valid_start", "valid_outcome", "runtime_signature_state", "training_runtime_signature_sha256", "terminal_marker_count", "state"}, "output_tree_audit.training_attempt")
    _assert_exact_keys(payload["deviation_ledger"], {"path", "sha256", "entry_count", "critical_or_high_count"}, "deviation_ledger")
    _assert_exact_keys(
        payload["paper_scale_numerical_result"],
        {"state", "value", "reason", "required_cutoff", "required_cycles", "current_qualification_cutoffs", "current_qualification_cycles", "published_anchor_comparison_eligible"},
        "paper_scale_numerical_result",
    )
    _assert_exact_keys(payload["external_claim_slots"], {"official_exact", "puviani_surpass", "paper_T_X_T_Y_T_Z_T_ch"}, "external_claim_slots")
    mutation_audit = _assert_exact_keys(payload["mutation_audit"], {"mutations", "mutation_count", "detected_count", "all_detected"}, "mutation_audit")
    if not isinstance(mutation_audit["mutations"], list):
        raise TypeError("mutation_audit.mutations must be a list")
    for row in mutation_audit["mutations"]:
        _assert_exact_keys(row, {"mutation_id", "target", "detected", "detection_reason"}, "mutation_audit.row")

    # Silence static linters while documenting that all action fields above are
    # subsequently value-bound in semantic validation.
    if not isinstance(action, Mapping):  # pragma: no cover
        raise AssertionError("unreachable")


def _validated_report_gpu_load_attestations(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    gpu_report = payload.get("gpu_load_attestations")
    if not isinstance(gpu_report, Mapping):
        raise ValueError("production report omitted GPU load attestations")
    config = payload["config"]
    implementation_hash = str(payload["implementation_sha256"])
    runtime_signature = payload["runtime_profile"]["signature"]
    training_values = gpu_report.get("training_launch_attestations")
    training_bindings = gpu_report.get("training_launch_bindings")
    if (
        not isinstance(training_values, list)
        or not training_values
        or not isinstance(training_bindings, list)
        or len(training_values) != len(training_bindings)
    ):
        raise ValueError("training GPU attestation census is empty or inconsistent")
    normalized_training: list[dict[str, Any]] = []
    normalized_bindings: list[dict[str, Any]] = []
    for value, binding_value in zip(
        training_values, training_bindings, strict=True
    ):
        normalized = _validate_gpu_load_attestation(
            value,
            config_sha256=_canonical_sha256(config),
            implementation_sha256=implementation_hash,
            expected_purpose="TRAINING_LAUNCH",
            current_runtime=runtime_signature,
            require_fresh=False,
            require_live_parent=False,
        )
        binding = _validated_gpu_attestation_binding(binding_value)
        if binding != _gpu_attestation_binding(normalized):
            raise ValueError("training GPU attestation binding mismatch")
        normalized_training.append(normalized)
        normalized_bindings.append(binding)
    hashes = [str(row["attestation_sha256"]) for row in normalized_bindings]
    if (
        hashes != sorted(hashes)
        or len(hashes) != len(set(hashes))
        or gpu_report.get("distinct_training_attestation_sha256") != hashes
    ):
        raise ValueError("training GPU attestation hash census drifted")
    finalizer = _validate_gpu_load_attestation(
        gpu_report.get("finalizer_launch_attestation"),
        config_sha256=_canonical_sha256(config),
        implementation_sha256=implementation_hash,
        expected_purpose="FINALIZER_LAUNCH",
        current_runtime=runtime_signature,
        require_fresh=False,
        require_live_parent=False,
    )
    finalizer_binding = _validated_gpu_attestation_binding(
        gpu_report.get("finalizer_launch_binding")
    )
    if finalizer_binding != _gpu_attestation_binding(finalizer):
        raise ValueError("finalizer GPU attestation binding mismatch")
    all_bindings = normalized_bindings + [finalizer_binding]
    same_target = (
        len({str(row["target_gpu_uuid"]).lower() for row in all_bindings}) == 1
        and len({str(row["target_gpu_name"]) for row in all_bindings}) == 1
        and len(
            {float(row["target_gpu_total_memory_mib"]) for row in all_bindings}
        )
        == 1
    )
    expected_scope = (
        "freshness and live supervisor-parent identity were enforced at each "
        "production process entry; report validation rechecks canonical/raw/device "
        "integrity without requiring historical attestations to remain unexpired"
    )
    if (
        gpu_report.get("schema_version")
        != "t9.1.3-gpu-load-attestation-report-v1"
        or gpu_report.get("all_forty_agents_bound") is not True
        or gpu_report.get("same_target_gpu_uuid_name_and_total_memory")
        is not True
        or not same_target
        or gpu_report.get("freshness_scope") != expected_scope
    ):
        raise ValueError("GPU load attestation report semantics drifted")
    return dict(gpu_report)


def _validate_report_semantics(
    payload: Mapping[str, Any], *, allow_missing_gpu_attestation: bool = False
) -> None:
    _validate_report_exact_schema(
        payload, allow_missing_gpu_attestation=allow_missing_gpu_attestation
    )
    try:
        datetime.fromisoformat(str(payload["generated_at_utc"]).replace("Z", "+00:00"))
        wall_time = float(payload["wall_time_seconds_finalize"])
    except (TypeError, ValueError) as error:
        raise ValueError("report runtime metadata is invalid") from error
    if not np.isfinite(wall_time) or wall_time < 0.0:
        raise ValueError("report finalization wall time is invalid")
    if payload.get("production_contract") is not True:
        raise ValueError("a PASS report must carry the frozen production contract")
    if not allow_missing_gpu_attestation:
        _validated_report_gpu_load_attestations(payload)
    if payload.get("namespace") != {
        "official_exact": {"state": "MISSING_EXTERNAL_ASSET", "value": None},
        "paper_constrained": "THIS_ARTIFACT",
        "project_native_directional": "T2.3.7_SEPARATE_NONTRANSFERABLE",
    }:
        raise ValueError("report evidence namespace drifted")
    if payload.get("claim_boundary") != REPORT_CLAIM_BOUNDARY:
        raise ValueError("report claim boundary was weakened")
    runtime_profile = payload["runtime_profile"]
    try:
        finalizer_signature, finalizer_runtime_hash = (
            _validated_training_runtime_signature(
                runtime_profile["signature"], label="report finalizer runtime"
            )
        )
        training_runtime = payload["training_runtime_contract"]
        training_signature, training_runtime_hash = (
            _validated_training_runtime_signature(
                training_runtime["signature"], label="report training runtime"
            )
        )
    except CorruptAgentArtifactError as error:
        raise ValueError(str(error)) from error
    try:
        _validate_production_determinism_signature(finalizer_signature)
        _validate_production_determinism_signature(training_signature)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    if (
        runtime_profile.get("schema_version")
        != "t9.1.3-finalization-runtime-profile-v1"
        or runtime_profile.get("role")
        != "FINALIZATION_AND_LIVE_VALIDATION_RUNTIME"
        or runtime_profile.get("signature_sha256") != finalizer_runtime_hash
        or runtime_profile.get("rng_contract")
        != (
            "explicit counter-derived simulator seeds; per-agent initialization seed; "
            "terminal torch CPU/CUDA and NumPy states retained"
        )
        or training_runtime.get("schema_version")
        != "t9.1.3-training-runtime-contract-v1"
        or training_runtime.get("signature_sha256") != training_runtime_hash
        or training_runtime.get("agent_signature_count") != 40
        or training_runtime.get("distinct_agent_signature_sha256")
        != [training_runtime_hash]
        or training_runtime.get("all_agents_identical") is not True
        or training_runtime.get("finalizer_signature_sha256")
        != finalizer_runtime_hash
        or training_runtime.get("equals_finalizer_runtime_signature") is not True
        or training_runtime_hash != finalizer_runtime_hash
        or training_signature != finalizer_signature
    ):
        raise ValueError("training/finalization runtime contract drifted")
    config = payload["config"]
    if payload.get("t9_1_1_binding") != _verify_parent_protocol(config):
        raise ValueError("report parent protocol binding drifted")
    deviation_path, _, deviations = _validated_deviation_ledger(config)
    expected_deviation_summary = {
        "path": _portable_artifact_path(deviation_path),
        "sha256": _file_sha256(deviation_path),
        "entry_count": len(deviations),
        "critical_or_high_count": sum(
            row.get("severity") in {"High", "Critical"} for row in deviations
        ),
    }
    if payload.get("deviation_ledger") != expected_deviation_summary:
        raise ValueError("report deviation-ledger summary drifted")
    if payload.get("protocol_transcription") != config["protocol_transcription"]:
        raise ValueError("report protocol transcription drifted from the frozen config")
    expected_action = {
        "project_order": list(PROJECT_ACTION_ORDER),
        "gqf_order": list(GQF_ACTION_ORDER),
        "permutation_roundtrip": True,
        "bounds": "14 residuals in [-2,2], virtual rotation in [-1,1] after tanh",
        "ranking_eligibility": config["ranking_eligibility"],
    }
    if payload.get("action_contract") != expected_action:
        raise ValueError("report action contract drifted")
    sources = payload["source_registry"]
    if (
        sources["paper"].get("doi") != PAPER_DOI
        or sources["paper"].get("arxiv") != PAPER_ARXIV
        or sources["official_gqf"].get("required_commit") != GQF_COMMIT
        or sources["official_gqf"].get("license") != "MIT"
        or sources.get("config_declared_source") != config["source_registry"]
    ):
        raise ValueError("report source registry claim fields drifted")
    selection = payload.get("selection_contract", {})
    if (
        selection.get("within_agent") != "max validation score; tie earlier epoch"
        or selection.get("across_agents")
        != "max best-validation score; tie earlier epoch then lower root seed"
        or selection.get("formal_used_for_selection") is not False
        or selection.get("test_used_for_selection") is not False
        or selection.get("paper_best_of_20_test_selection_replaced") is not True
        or selection.get("ledger_path")
        != "docs/t9_1_3_puviani_selection_ledger.csv"
    ):
        raise ValueError("report selection contract drifted")
    selection_warning = (
        "best-of-20 agents and best-of-validation-checkpoints can be optimistic; "
        "these validation scores are selection diagnostics, not an unbiased performance "
        "estimate or a T9.1.3 performance gate; six-state qualification seeds are held out"
    )
    for strategy in ("mf", "nmf"):
        item = selection["selected"][strategy]
        distributions = (
            item["all_agent_best_validation_score_distribution"],
            item["all_agent_initial_validation_score_distribution"],
            item["all_agent_initial_to_best_gain_distribution"],
            item["all_agent_selected_checkpoint_epoch_distribution"],
        )
        if (
            item.get("agent_count") != 20
            or item.get("validation_candidates_per_agent") != 26
            or item.get("total_agent_checkpoint_candidates") != 520
            or item.get("selection_optimism_warning") != selection_warning
            or not all(
                _distribution_payload_valid(value, count=20)
                for value in distributions
            )
            or float(
                item["all_agent_initial_to_best_gain_distribution"]["minimum"]
            )
            < -1.0e-15
        ):
            raise ValueError("selection multiplicity/optimism disclosure drifted")
    population = payload.get("population", {})
    if (
        population.get("paired_root_count") != 20
        or population.get("total_agent_count") != 40
        or population.get("agents_per_family") != {"mf": 20, "nmf": 20}
        or population.get("all_agents_published_not_only_winner") is not True
        or population.get("agent_registry")
        != "docs/t9_1_3_puviani_agent_registry.csv"
    ):
        raise ValueError("report population census drifted")
    training = payload.get("training_ledger", {})
    if (
        training.get("path") != "docs/t9_1_3_puviani_training_ledger.parquet"
        or training.get("rows") != 40_000
        or training.get("all_epochs_all_agents") is not True
        or not _sentinel_retrain_audit_valid(
            training.get("family_sentinel_full_retrain"),
            expected_epochs=1_000,
            expected_population_agents=40,
            expected_roots={
                "mf": int(config["training"]["paired_root_seeds"][0]),
                "nmf": int(config["training"]["paired_root_seeds"][0]),
            },
        )
    ):
        raise ValueError("report training census drifted")
    evaluator = payload.get("six_state_evaluator", {})
    if (
        evaluator.get("state_labels") != list(STATE_LABELS)
        or evaluator.get("methods") != ["standard", "mf", "nmf"]
        or evaluator.get("trajectory_rows") != 1_008
        or evaluator.get("event_rows") != 20_160
        or evaluator.get("trajectory_path")
        != "docs/t9_1_3_puviani_six_state_trajectories.parquet"
        or evaluator.get("event_path")
        != "docs/t9_1_3_puviani_six_state_events.parquet"
        or evaluator.get("postselection") is not False
        or evaluator.get("all_trajectories_retained") is not True
        or evaluator.get("development_only") is not True
        or evaluator.get("state_definition_id")
        != "LOWDIN_ORTHONORMALIZED_FINITE_CUTOFF_GRID_CODE_V1"
        or evaluator.get("logical_operator_definition_id")
        != "PROJECTOR_DEFINED_CODE_PAULI_V1"
        or not str(evaluator.get("monte_carlo_design", "")).startswith(
            "paired_common_random_numbers"
        )
        or not _six_state_raw_replay_audit_valid(
            config,
            evaluator.get("raw_replay_audit", {}),
            trajectory_count=1_008,
            event_count=20_160,
        )
        or evaluator.get("qualification_scope")
        != (
            "finite-cutoff ten-cycle executable/raw qualification only; not a "
            "paper lifetime and not Phase9 >=1e4-cycle formal evidence"
        )
        or not _empirical_reconstruction_valid(evaluator.get("reconstruction", {}))
    ):
        raise ValueError("six-state report semantics or census drifted")
    for methods in evaluator["reconstruction"].values():
        for diagnostic in methods.values():
            if (
                diagnostic.get("paper_lifetime_value") is not None
                or diagnostic.get("paper_lifetime_state") != PAPER_NUMERIC_STATE
            ):
                raise ValueError("paper lifetime was injected into qualification")
    numeric = payload.get("paper_scale_numerical_result", {})
    if (
        numeric.get("state") != PAPER_NUMERIC_STATE
        or numeric.get("value") is not None
        or numeric.get("reason") != PAPER_NUMERIC_REASON
        or numeric.get("published_anchor_comparison_eligible") is not False
        or numeric.get("required_cutoff") != 100
        or numeric.get("required_cycles") != 1000
        or numeric.get("current_qualification_cutoffs") != [12, 16]
        or numeric.get("current_qualification_cycles") != 10
    ):
        raise ValueError("paper-scale numerical boundary drifted")
    external = payload.get("external_claim_slots", {})
    if external != {
        "official_exact": None,
        "puviani_surpass": None,
        "paper_T_X_T_Y_T_Z_T_ch": None,
    }:
        raise ValueError("external/Puviani claim slot was illegally populated")
    failure_ledger = payload["failure_attempt_ledger"]
    if (
        failure_ledger.get("count") != len(failure_ledger["attempts"])
        or failure_ledger.get("distinct_runtime_signature_sha256")
        != sorted(
            {
                str(row["training_runtime_signature_sha256"])
                for row in failure_ledger["attempts"]
            }
        )
        or failure_ledger.get("all_failure_runtime_signatures_self_bound")
        is not True
        or failure_ledger.get("silent_drop") is not False
        or failure_ledger.get("note")
        != (
            "A retained failed attempt does not replace the required completed agent; "
            "all 40 completed checkpoints remain mandatory."
        )
    ):
        raise ValueError("failure-attempt report contract drifted")
    output_audit = payload["output_tree_audit"]
    if (
        output_audit.get("root") != "docs/t9_1_3_puviani_agents"
        or output_audit.get("training_attempt_count")
        != len(output_audit["training_attempts"])
        or output_audit.get("unknown_or_orphan_files") != []
        or output_audit.get("all_training_attempts_terminal_and_valid") is not True
        or output_audit.get("every_agent_has_successful_attempt") is not True
    ):
        raise ValueError("output-tree audit contract drifted")


def _report_output_dir_for_lock(payload: Mapping[str, Any]) -> Path:
    """Resolve the untrusted report's namespace without permitting lock escape."""

    output_audit = payload.get("output_tree_audit")
    if not isinstance(output_audit, Mapping):
        raise TypeError("report output-tree audit is required for live locking")
    raw_root = output_audit.get("root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError("report output-tree root is invalid")
    repository = Path(__file__).resolve().parents[2]
    output_root = _resolve_repo_path(raw_root).resolve()
    try:
        output_root.relative_to(repository.resolve())
    except ValueError as error:
        raise ValueError("report output-tree lock escaped the repository") from error
    return output_root


def _live_publication_snapshot(
    manifest: Mapping[str, Any], output_root: Path
) -> str:
    """Hash a stable live-file census while the namespace lock is held."""

    repository = Path(__file__).resolve().parents[2]
    manifest_rows: list[dict[str, Any]] = []
    for entry in manifest.get("files", []):
        path = (repository / str(entry["path"])).resolve()
        try:
            path.relative_to(repository.resolve())
        except ValueError as error:
            raise ValueError("artifact manifest escaped the repository root") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing published artifact: {path}")
        row = {
            "path": path.relative_to(repository.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if row != {
            "path": entry["path"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }:
            raise ValueError(f"published artifact changed: {path}")
        manifest_rows.append(row)
    output_rows = [
        {
            "path": path.resolve().relative_to(repository.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and not _is_canonical_finalize_lock_management_file(path, output_root)
    ]
    return _canonical_sha256(
        {"manifest_files": manifest_rows, "output_tree_files": output_rows}
    )


def _assert_live_publication_snapshot_unchanged(
    manifest: Mapping[str, Any], output_root: Path, initial_snapshot: str
) -> None:
    if _live_publication_snapshot(manifest, output_root) != initial_snapshot:
        raise ValueError(
            "live publication manifest/output census changed during validation"
        )


def _validate_report_locked(
    payload_or_path: Mapping[str, Any] | str | Path,
    *,
    verify_live_files: bool = True,
    _require_mutation_audit: bool = True,
    _structural_test_only: bool = False,
) -> dict[str, Any]:
    """Validate with the caller already owning the namespace lock for live reads."""

    if not verify_live_files and not _structural_test_only:
        raise ValueError(
            "non-live validation is internal structural-test scope, not a PASS channel"
        )
    if isinstance(payload_or_path, Mapping):
        payload = dict(payload_or_path)
    else:
        payload = _load_json(Path(payload_or_path))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("task_id") != TASK_ID:
        raise ValueError("T9.1.3 report identity mismatch")
    claimed_hash = payload.get("analysis_sha256")
    unhashed = dict(payload)
    unhashed.pop("analysis_sha256", None)
    if claimed_hash != _canonical_sha256(unhashed):
        raise ValueError("T9.1.3 analysis hash mismatch")
    gates = payload.get("gates")
    required = payload.get("required_gates")
    if not isinstance(gates, dict) or not isinstance(required, list):
        raise TypeError("report gate structure is invalid")
    if tuple(required) != REQUIRED_GATE_NAMES or tuple(gates) != REQUIRED_GATE_NAMES:
        raise ValueError("required T9.1.3 gate registry drifted")
    if not all(gates.get(name) is True for name in REQUIRED_GATE_NAMES):
        raise ValueError("one or more required T9.1.3 gates failed")
    if payload.get("status") != STATUS_PASS:
        raise ValueError("artifact lane status is not PASS")
    if payload.get("evidence_grade") != EVIDENCE_GRADE:
        raise ValueError("evidence grade escalation detected")
    embedded_config = payload.get("config")
    if not isinstance(embedded_config, dict):
        raise TypeError("sealed report must embed its complete experiment config")
    if payload.get("config_sha256") != _canonical_sha256(embedded_config):
        raise ValueError("embedded config hash mismatch")
    _validate_config(embedded_config, production=True)
    if verify_live_files:
        _validate_runtime_for_config(embedded_config, production=True)
    _validate_report_semantics(
        payload,
        allow_missing_gpu_attestation=(
            _structural_test_only and "gpu_load_attestations" not in payload
        ),
    )
    if payload.get("implementation_sha256") != implementation_sha256():
        raise ValueError("live implementation hash differs from the sealed report")
    manifest, manifested_paths = _validated_manifest_paths(payload)
    mutation_audit = payload.get("mutation_audit")
    if _require_mutation_audit:
        if not isinstance(mutation_audit, dict):
            raise ValueError("semantic mutation audit is mandatory")
        mutations = mutation_audit.get("mutations")
        expected_mutation_ids = tuple(
            [f"published_gate_false:{name}" for name in REQUIRED_GATE_NAMES]
            + list(SOURCE_EVIDENCE_MUTATION_IDS)
            + list(CLAIM_MUTATION_IDS)
        )
        if (
            not isinstance(mutations, list)
            or tuple(row.get("mutation_id") for row in mutations)
            != expected_mutation_ids
            or any(row.get("detected") is not True for row in mutations)
            or mutation_audit.get("mutation_count") != len(expected_mutation_ids)
            or mutation_audit.get("detected_count") != len(expected_mutation_ids)
            or mutation_audit.get("all_detected") is not True
        ):
            raise ValueError("semantic mutation audit is incomplete or drifted")
    if verify_live_files:
        root = Path(__file__).resolve().parents[2]
        output_audit = payload.get("output_tree_audit", {})
        if output_audit.get("root") != "docs/t9_1_3_puviani_agents":
            raise ValueError("sealed output root is not the canonical production root")
        output_root = root / str(output_audit["root"])
        if not output_root.is_dir():
            raise FileNotFoundError(f"missing output artifact tree: {output_root}")
        actual_output_paths = {
            path.resolve().relative_to(root.resolve()).as_posix()
            for path in output_root.rglob("*")
            if path.is_file()
            and not _is_canonical_finalize_lock_management_file(path, output_root)
        }
        output_prefix = output_root.resolve().relative_to(root.resolve()).as_posix() + "/"
        manifested_output_paths = {
            path for path in manifested_paths if path.startswith(output_prefix)
        }
        if actual_output_paths != manifested_output_paths:
            raise ValueError("live output tree differs from the sealed filesystem census")
        expected_manifest_paths = set(MANDATORY_SHARED_ARTIFACT_PATHS) | actual_output_paths
        if manifested_paths != expected_manifest_paths:
            raise ValueError("manifest contains missing or extra publication paths")
        initial_publication_snapshot = _live_publication_snapshot(
            manifest, output_root
        )
        actual_audit = _audit_output_tree(
            output_dir=output_root,
            config=embedded_config,
            implementation_hash=str(payload["implementation_sha256"]),
        )
        report_attempts = output_audit.get("training_attempts")
        if (
            output_audit.get("output_file_count") != actual_audit["output_file_count"]
            or output_audit.get("unknown_or_orphan_files")
            != actual_audit["unknown_or_orphan_files"]
            or output_audit.get("training_attempt_count")
            != len(actual_audit["training_attempts"])
            or report_attempts != actual_audit["training_attempts"]
            or output_audit.get("all_training_attempts_terminal_and_valid")
            is not True
            or actual_audit["all_training_attempts_terminal_and_valid"] is not True
            or output_audit.get("every_agent_has_successful_attempt") is not True
            or actual_audit["every_agent_has_successful_attempt"] is not True
        ):
            raise ValueError("live training-attempt/output census drifted")
        failure_ledger = payload.get("failure_attempt_ledger", {})
        if (
            failure_ledger.get("count") != len(actual_audit["failure_attempts"])
            or failure_ledger.get("attempts") != actual_audit["failure_attempts"]
            or failure_ledger.get("distinct_runtime_signature_sha256")
            != sorted(
                {
                    str(row["training_runtime_signature_sha256"])
                    for row in actual_audit["failure_attempts"]
                }
            )
            or failure_ledger.get("all_failure_runtime_signatures_self_bound")
            is not True
            or failure_ledger.get("silent_drop") is not False
        ):
            raise ValueError("live failure-attempt ledger drifted")

        with (root / "docs/t9_1_3_puviani_agent_registry.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            agent_rows = list(csv.DictReader(handle))
        with (root / "docs/t9_1_3_puviani_selection_ledger.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            selection_rows = list(csv.DictReader(handle))
        if len(agent_rows) != 40 or len(selection_rows) != 1_040:
            raise ValueError("live CSV ledger row census drifted")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError as error:  # pragma: no cover - DLEnv runtime.
            raise RuntimeError("pyarrow is required to validate T9.1.3 Parquet ledgers") from error
        training_path = root / "docs/t9_1_3_puviani_training_ledger.parquet"
        trajectory_path = root / "docs/t9_1_3_puviani_six_state_trajectories.parquet"
        event_path = root / "docs/t9_1_3_puviani_six_state_events.parquet"
        training_table = pq.read_table(training_path)
        trajectory_table = pq.read_table(trajectory_path)
        event_table = pq.read_table(event_path)
        if (
            training_table.num_rows != 40_000
            or trajectory_table.num_rows != 1_008
            or event_table.num_rows != 20_160
            or trajectory_table.column_names != list(TRAJECTORY_ROW_KEYS)
            or event_table.column_names != list(EVENT_ROW_KEYS)
        ):
            raise ValueError("live Parquet row or schema census drifted")

        # Rebuild every derived ledger and diagnostic from the immutable raw
        # agent bundles/rows.  A re-sealed report cannot substitute summaries.
        live_parent = _verify_parent_protocol(embedded_config)
        if payload.get("t9_1_1_binding") != live_parent:
            raise ValueError("live T9.1.1 parent binding drifted")
        live_sources = _source_registry(embedded_config)
        if payload.get("source_registry") != live_sources:
            raise ValueError("live paper/GQF source registry drifted")
        live_agents, live_implementation = _all_agent_records(
            embedded_config, output_root
        )
        if live_implementation != payload["implementation_sha256"]:
            raise ValueError("agent population implementation hash drifted")
        live_runtime_profile = _runtime_profile()
        live_training_runtime_contract = _training_runtime_contract(
            live_agents, live_runtime_profile, production=True
        )
        if (
            payload.get("runtime_profile") != live_runtime_profile
            or payload.get("training_runtime_contract")
            != live_training_runtime_contract
        ):
            raise ValueError("live training/finalization runtime contract drifted")
        sealed_gpu_report = payload.get("gpu_load_attestations")
        if not isinstance(sealed_gpu_report, Mapping):
            raise ValueError("sealed report omitted GPU attestation evidence")
        live_gpu_report = _gpu_load_attestation_report(
            live_agents,
            sealed_gpu_report.get("finalizer_launch_attestation"),
            config=embedded_config,
            implementation_hash=live_implementation,
            runtime_signature=live_runtime_profile["signature"],
        )
        if live_gpu_report != sealed_gpu_report:
            raise ValueError(
                "GPU load attestation report is not derivable from live agents"
            )
        live_selected, expected_selection_rows = _select_agents(live_agents)
        expected_agent_rows = _agent_registry_rows(live_agents, live_selected)
        expected_training_rows = _training_rows(live_agents)
        if agent_rows != _csv_roundtrip_rows(expected_agent_rows):
            raise ValueError("agent registry is not derivable from live checkpoints")
        if selection_rows != _csv_roundtrip_rows(expected_selection_rows):
            raise ValueError("selection ledger is not derivable from validation records")
        if payload["selection_contract"].get(
            "selection_lock_sha256"
        ) != _canonical_sha256(expected_selection_rows):
            raise ValueError("selection lock is not derivable from validation records")
        expected_training_table = pa.Table.from_pylist(expected_training_rows)
        if not training_table.equals(expected_training_table):
            raise ValueError("training ledger is not derivable from live checkpoints")

        trajectory_rows_live = trajectory_table.to_pylist()
        event_rows_live = event_table.to_pylist()
        live_reconstruction = _reconstruct_qualification_from_rows(
            embedded_config, trajectory_rows_live
        )
        if (
            payload["six_state_evaluator"].get("reconstruction")
            != live_reconstruction
        ):
            raise ValueError("six-state reconstruction is not derivable from raw rows")
        if not _event_timeline_valid(
            event_rows_live,
            trajectory_rows_live,
            half_cycles=2
            * int(embedded_config["six_state_qualification"]["full_cycles"]),
        ):
            raise ValueError("raw event rows violate the paper-causal timeline")
        live_raw_replay_audit = _six_state_raw_replay_audit(
            embedded_config,
            live_selected,
            trajectory_rows_live,
            event_rows_live,
        )
        if not _six_state_raw_replay_audits_compatible(
            embedded_config,
            payload["six_state_evaluator"].get("raw_replay_audit", {}),
            live_raw_replay_audit,
            trajectory_count=trajectory_table.num_rows,
            event_count=event_table.num_rows,
        ):
            raise ValueError(
                "six-state raw rows are not reproducible from selected checkpoints, "
                "RNG and the seeded same-runtime/same-backend physics replay"
            )

        selected_summary = _selection_summary(live_agents, live_selected)
        if payload["selection_contract"].get("selected") != selected_summary:
            raise ValueError("selected-agent summary is not derivable from validation")

        deviation_path, deviation_payload, deviations = _validated_deviation_ledger(
            embedded_config
        )
        expected_deviation_summary = {
            "path": deviation_path.relative_to(root).as_posix(),
            "sha256": _file_sha256(deviation_path),
            "entry_count": len(deviations),
            "critical_or_high_count": sum(
                row.get("severity") in {"High", "Critical"}
                for row in deviations
            ),
        }
        if payload.get("deviation_ledger") != expected_deviation_summary:
            raise ValueError("live deviation-ledger summary drifted")

        recomputed_gates = _build_gates(
            config=embedded_config,
            parent=live_parent,
            sources=live_sources,
            deviations=deviations,
            agents=live_agents,
            agent_rows=expected_agent_rows,
            selection_rows=expected_selection_rows,
            trajectory_rows=trajectory_rows_live,
            event_rows=event_rows_live,
            training_rows=expected_training_rows,
            sentinel_retrain_audit=payload["training_ledger"][
                "family_sentinel_full_retrain"
            ],
            reconstruction=live_reconstruction,
            raw_replay_audit=live_raw_replay_audit,
            manifest=manifest,
            failure_attempts=actual_audit["failure_attempts"],
            implementation_hash=live_implementation,
            training_runtime_contract=live_training_runtime_contract,
            output_audit=actual_audit,
            expected_manifest_file_count=(
                len(MANDATORY_SHARED_ARTIFACT_PATHS)
                + actual_audit["output_file_count"]
            ),
        )
        if recomputed_gates != payload["gates"] or not all(
            recomputed_gates.values()
        ):
            raise ValueError("published gates are not derivable from live evidence")
        rerun_mutation_audit = _mutation_audit(payload)
        if rerun_mutation_audit != payload.get("mutation_audit"):
            raise ValueError("semantic mutation audit was not reproducible")
        _assert_live_publication_snapshot_unchanged(
            manifest, output_root, initial_publication_snapshot
        )
    return payload


def validate_report(
    payload_or_path: Mapping[str, Any] | str | Path,
    *,
    verify_live_files: bool = True,
    _require_mutation_audit: bool = True,
    _structural_test_only: bool = False,
) -> dict[str, Any]:
    """Acquire the publication lock and validate one stable report snapshot."""

    if not verify_live_files:
        return _validate_report_locked(
            payload_or_path,
            verify_live_files=False,
            _require_mutation_audit=_require_mutation_audit,
            _structural_test_only=_structural_test_only,
        )
    is_mapping = isinstance(payload_or_path, Mapping)
    report_path = None if is_mapping else Path(payload_or_path)
    preview = (
        copy.deepcopy(dict(payload_or_path))
        if is_mapping
        else _load_json(report_path)
    )
    output_root = _report_output_dir_for_lock(preview)
    with _namespace_finalize_lock(output_root):
        locked_payload = copy.deepcopy(preview) if is_mapping else _load_json(report_path)
        if _report_output_dir_for_lock(locked_payload) != output_root:
            raise ValueError("report output-tree root changed during lock acquisition")
        result = _validate_report_locked(
            locked_payload,
            verify_live_files=True,
            _require_mutation_audit=_require_mutation_audit,
            _structural_test_only=_structural_test_only,
        )
        if report_path is not None and _load_json(report_path) != locked_payload:
            raise ValueError("report changed during locked live validation")
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase9/t9_1_3_puviani_paper_constrained.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/t9_1_3_puviani_agents")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/t9_1_3_puviani_paper_constrained.json")
    )
    parser.add_argument(
        "--agent-registry", type=Path, default=Path("docs/t9_1_3_puviani_agent_registry.csv")
    )
    parser.add_argument(
        "--selection-ledger", type=Path, default=Path("docs/t9_1_3_puviani_selection_ledger.csv")
    )
    parser.add_argument(
        "--training-ledger", type=Path, default=Path("docs/t9_1_3_puviani_training_ledger.parquet")
    )
    parser.add_argument(
        "--trajectories", type=Path, default=Path("docs/t9_1_3_puviani_six_state_trajectories.parquet")
    )
    parser.add_argument(
        "--events", type=Path, default=Path("docs/t9_1_3_puviani_six_state_events.parquet")
    )
    parser.add_argument(
        "--gpu-attestation",
        type=Path,
        default=None,
        help="fresh supervisor-sealed GPU load attestation (mandatory in production)",
    )
    parser.add_argument(
        "--supervisor-nmf-release-gate",
        type=Path,
        default=None,
        help=(
            "private production-supervisor gate: NMF validates and waits here "
            "before touching the output namespace"
        ),
    )
    parser.add_argument(
        "--supervisor-nmf-release-nonce",
        default=None,
        help="private nonce binding the NMF waiter to its supervisor release",
    )
    parser.add_argument(
        "--supervisor-training-deadline-utc",
        default=None,
        help="absolute UTC deadline shared by the serialized MF/NMF transaction",
    )
    parser.add_argument("--family", choices=("mf", "nmf", "all"), default="all")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    config = _load_json(arguments.config)
    if arguments.pilot:
        if _canonical_sha256(config) == PRODUCTION_CONFIG_SHA256:
            raise SystemExit(
                "--pilot refuses the frozen 40x1000 production config; provide an "
                "explicit reduced pilot config so the workload cannot be mistaken"
            )
        if not arguments.train_only:
            raise SystemExit(
                "pilot execution is train-only and cannot emit a production PASS seal"
            )
    production = not arguments.pilot
    serial_values = (
        arguments.supervisor_nmf_release_gate,
        arguments.supervisor_nmf_release_nonce,
        arguments.supervisor_training_deadline_utc,
    )
    serial_requested = any(value is not None for value in serial_values)
    if serial_requested and not all(value is not None for value in serial_values):
        raise SystemExit("all supervisor NMF serial-release arguments are required")
    if serial_requested and (
        not production
        or arguments.family != "nmf"
        or not arguments.train_only
        or arguments.finalize_only
        or arguments.gpu_attestation is None
        or _canonical_sha256(config) != PRODUCTION_CONFIG_SHA256
    ):
        raise SystemExit(
            "supervisor NMF serial-release arguments require the canonical "
            "production NMF-only train-only launch with a live GPU attestation"
        )
    if (
        production
        and arguments.family == "nmf"
        and arguments.train_only
        and not serial_requested
    ):
        raise SystemExit(
            "canonical production NMF-only training requires the supervisor "
            "MF-success serial-release gate"
        )
    result: dict[str, Any] | None = None
    if not arguments.finalize_only:
        result = train_population(
            config,
            output_dir=arguments.output_dir,
            family=arguments.family,
            production=production,
            gpu_attestation=arguments.gpu_attestation,
            serial_release_gate=arguments.supervisor_nmf_release_gate,
            serial_release_nonce=arguments.supervisor_nmf_release_nonce,
            serial_release_deadline_utc=arguments.supervisor_training_deadline_utc,
        )
        print(json.dumps(result, indent=2), flush=True)
    if not arguments.train_only:
        report = finalize_artifacts(
            config,
            output_dir=arguments.output_dir,
            report_path=arguments.report,
            agent_registry_path=arguments.agent_registry,
            selection_ledger_path=arguments.selection_ledger,
            training_ledger_path=arguments.training_ledger,
            trajectory_path=arguments.trajectories,
            event_path=arguments.events,
            production=production,
            gpu_attestation=arguments.gpu_attestation,
        )
        validate_report(report, verify_live_files=True)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "analysis_sha256": report["analysis_sha256"],
                    "paper_scale_numerical_result": report[
                        "paper_scale_numerical_result"
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if report["status"] == STATUS_PASS else 2
    return 0 if result is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVIDENCE_GRADE",
    "GQF_ACTION_ORDER",
    "PAPER_NUMERIC_STATE",
    "PROJECT_ACTION_ORDER",
    "SCHEMA_VERSION",
    "STATUS_PASS",
    "finalize_artifacts",
    "gqf_to_project_action",
    "implementation_sha256",
    "project_to_gqf_action",
    "train_population",
    "validate_report",
]
