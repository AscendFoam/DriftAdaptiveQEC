"""T6.17.3 read-only learned-model eligibility and replay audit.

The audit deliberately separates direct decoders, causal estimators and
RL/NMF physical controllers.  It never trains, tunes, or reselects a model.
Only checkpoints matching the complete single-mode decoder task signature
could receive LER/timing ranking values; all others remain explicit nulls.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from cnn_fpga.model.tiny_cnn import predict_from_artifact


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.17.3"
SCHEMA_VERSION = "t6.17.3-learned-model-eligibility-replay-v1"
VERDICT = "PASS_READONLY_LEARNED_ELIGIBILITY_NO_SAME_TASK_CHECKPOINT"
PREREG_CONFIG = ROOT / "configs" / "literature" / "t6_16_3_secondary_preregistration.json"
ONTOLOGY = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
SOURCE_AUDIT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
T6155 = ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate.json"
T514 = ROOT / "docs" / "t5_1_4_algorithm_branch_verdict.json"
T543 = ROOT / "docs" / "t5_4_3_causal_ablation_negative_results.json"
T327_REPORT = ROOT / "docs" / "t3_2_7_latest_outcome_markovian_validation.json"
T327_CHECKPOINT = ROOT / "docs" / "t3_2_7_latest_outcome_markovian_checkpoints.pt"
T3210_REPORT = ROOT / "docs" / "t3_2_10_exponential_recurrence_validation.json"
T3210_CHECKPOINT = ROOT / "docs" / "t3_2_10_exponential_recurrence_checkpoints.pt"
T411_REPORT = ROOT / "docs" / "t4_1_1_slow_loop_model_selection_validation.json"
T411_CHECKPOINT = ROOT / "docs" / "t4_1_1_slow_loop_model_selection_checkpoints.pt"
T415_REPORT = ROOT / "docs" / "t4_1_5_teacher_student_validation.json"
T415_ARTIFACT = ROOT / "docs" / "t4_1_5_distilled_student_checkpoint.json"
T441_CHECKPOINT = ROOT / "docs" / "t4_4_1_bounded_residual_rnn_teacher_checkpoints.pt"
T443_ARTIFACT = ROOT / "docs" / "t4_4_3_low_dimensional_student.json"
T443_CHECKPOINT = ROOT / "docs" / "t4_4_3_low_dimensional_student_candidates.pt"
T545_CHECKPOINT = ROOT / "docs" / "t5_4_5_horizon_extrapolation_candidates.pt"
T554_REPORT = ROOT / "docs" / "t5_5_4_gru_student_hardware_feasibility.json"
T554_MANIFEST = ROOT / "cnn_fpga" / "rtl" / "generated" / "t5_5_4_quantized_gru_manifest.json"
T237_CHECKPOINT = ROOT / "docs" / "t2_3_7_nmf_directional_ranking_checkpoints.pt"
GQF_INTAKE = ROOT / "docs" / "t6_8_3_gqf_official_intake.json"
LEGACY_MANIFEST = ROOT / "artifacts" / "datasets" / "runtime_b_residual_v1" / "manifest.json"
LEGACY_INPUT = ROOT / "artifacts" / "datasets" / "runtime_b_residual_v1" / "test.npz"
LEGACY_MODEL = ROOT / "artifacts" / "models" / "runtime_b_residual_v1" / "tiny_cnn_20260401_083648_2fc740424c0d.npz"
LEGACY_EVAL = ROOT / "artifacts" / "reports" / "runtime_b_residual_v1" / "eval_test_20260401_083649.json"
STATIC_THETA_MODEL = ROOT / "artifacts" / "models" / "static_theta_v2" / "tiny_cnn_20260319_151717_b87c6c227b57.npz"
DEFAULT_REPORT = ROOT / "docs" / "t6_17_3_learned_model_eligibility_replay.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_17_3_learned_model_eligibility_replay_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "learned_model_eligibility_replay.md"

METRICS = ["p_L", "p_X", "p_Y", "p_Z", "average_ler", "latency_ns", "macs_per_update"]
REQUIRED_BUDGET_FIELDS = ["MAC", "state_bytes", "workspace_bytes", "wall_clock", "precision", "cadence", "warmup"]
SIGNATURE_FIELDS = [
    "code_family", "modes_or_distance", "decision_target", "input_semantics",
    "history_horizon", "output_action", "noise_model", "observability",
    "online_privilege", "time_basis", "compute_budget", "precision", "evidence_level",
]
REQUIRED_SIGNATURE = {
    "code_family": "single-mode square approximate GKP",
    "modes_or_distance": "one oscillator",
    "decision_target": "per-round logical coset/correction and typed safety decision",
    "input_semantics": "frozen parent q/p syndrome code and registered observed history",
    "history_horizon": "registered causal per-round/window horizon",
    "output_action": "logical Pauli/frame correction plus typed safety action",
    "noise_model": "frozen parent secondary replay trace",
    "observability": "observed-only",
    "online_privilege": "causal inference without future or hidden truth",
    "time_basis": "every registered decoded round",
    "compute_budget": "matched registered production decoder budget",
    "precision": "production fixed-point",
    "evidence_level": "project checkpoint replay",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ontology_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("ontology", "source_metric_crosswalk", "ranking_policy", "parent_contracts", "verdict")
    }


def _source_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("scope", "sources", "methods", "claim_audit", "derived_evidence", "comparison_policy", "verdict")
    }


def _preregistered_experiment() -> dict[str, Any]:
    rows = [row for row in _load(PREREG_CONFIG)["experiments"] if row["task_id"] == TASK_ID]
    if len(rows) != 1:
        raise ValueError("T6.17.3 requires exactly one frozen preregistration row")
    return rows[0]


def _budget(
    mac: int | None,
    state_bytes: int | None,
    workspace_bytes: int | None,
    wall_clock: Any,
    precision: str | None,
    cadence: str | None,
    warmup: str | None,
    *,
    provenance: str,
) -> dict[str, Any]:
    return {
        "MAC": mac,
        "state_bytes": state_bytes,
        "workspace_bytes": workspace_bytes,
        "wall_clock": wall_clock,
        "precision": precision,
        "cadence": cadence,
        "warmup": warmup,
        "provenance": provenance,
    }


def _null_metrics(reason: str) -> dict[str, Any]:
    return {
        metric: {"value": None, "value_state": "N_A_NOT_APPLICABLE", "ranking_eligible": False, "reason": reason}
        for metric in METRICS
    }


def _source_method(source_id: str) -> dict[str, Any]:
    rows = [row for row in _load(SOURCE_AUDIT)["methods"] if row["source_id"] == source_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one source method for {source_id}")
    return rows[0]


def _candidate(
    candidate_id: str,
    category: str,
    native_lane: str,
    artifact: Path | None,
    member_count: int,
    actual_signature: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    source_id: str | None = None,
    replay_state: str = "NOT_REPLAYED_INELIGIBLE",
) -> dict[str, Any]:
    if set(actual_signature) != set(SIGNATURE_FIELDS):
        raise ValueError(f"{candidate_id} signature is incomplete")
    if set(budget) != set(REQUIRED_BUDGET_FIELDS) | {"provenance"}:
        raise ValueError(f"{candidate_id} budget is incomplete")
    field_checks = {
        field: {"required": REQUIRED_SIGNATURE[field], "actual": actual_signature[field], "match": actual_signature[field] == REQUIRED_SIGNATURE[field]}
        for field in SIGNATURE_FIELDS
    }
    mismatches = [field for field, row in field_checks.items() if not row["match"]]
    eligible = not mismatches
    if eligible and artifact is None:
        raise ValueError("eligible checkpoint must have a local artifact")
    return {
        "candidate_id": candidate_id,
        "category": category,
        "native_lane": native_lane,
        "source_id": source_id,
        "artifact": _binding(artifact) if artifact is not None else None,
        "member_count": member_count,
        "signature": dict(actual_signature),
        "signature_checks": field_checks,
        "mismatch_fields": mismatches,
        "same_task_eligible": eligible,
        "eligibility_state": "PROJECT_NATIVE_MATCHED" if eligible else "INELIGIBLE_TASK_SIGNATURE",
        "budget": dict(budget),
        "replay_state": replay_state if not eligible else "PENDING_ELIGIBLE_REPLAY",
        "metrics": _null_metrics("INELIGIBLE_TASK_SIGNATURE") if not eligible else {},
    }


def _signature(**updates: Any) -> dict[str, Any]:
    signature = dict(REQUIRED_SIGNATURE)
    signature.update(updates)
    return signature


def _candidate_registry(diagnostic_wall_clock: Mapping[str, Any]) -> list[dict[str, Any]]:
    t327 = _load(T327_REPORT)
    t3210 = _load(T3210_REPORT)
    t411 = _load(T411_REPORT)
    profiles = t411["resource_profiles"]
    t415 = _load(T415_REPORT)
    t443 = _load(T443_ARTIFACT)
    t554 = _load(T554_REPORT)
    student = next(row for row in t554["candidates"] if row["candidate_id"] == "distilled_student_q3_14_state4_serial")
    quantized = next(row for row in t554["candidates"] if row["candidate_id"] == "quantized_gru_int8_q14_lower_bound")
    source_rows = {source_id: _source_method(source_id) for source_id in ("WANG2022_DIRECT_NN", "SIVAK2023_RL_GKP", "SIVAK2026_RL_DRIFT", "PUVIANI2025_NMF")}
    legacy_workspace = 21 * 3 * 3 * 32 * 32 * 8
    legacy_macs = 32 * 32 * 12 * 21 * 3 * 3 + 3072 * 64 + 64 * 2
    controller_signature = _signature(
        decision_target="history/performance to physical-control update",
        input_semantics="protocol-specific observations or performance",
        history_horizon="protocol-specific controller history",
        output_action="continuous physical-control parameters",
        time_basis="per control half-cycle/epoch",
        compute_budget="controller-specific unmatched budget",
        precision="source/project controller precision",
    )
    rows = [
        _candidate(
            "legacy_residual_tinycnn", "legacy_cnn_parameter_estimator", "single_mode_decoder", LEGACY_MODEL, 1,
            _signature(
                decision_target="five-window residual b_q/b_p parameter regression",
                input_semantics="21-channel 5x32-cycle histograms plus teacher parameters/deltas",
                history_horizon="five 32-cycle windows with edge-repeat warmup",
                output_action="two continuous b_q/b_p residual estimates",
                noise_model="four legacy static/ramp/step/periodic parameter scenarios",
                time_basis="one update per histogram window",
                compute_budget="unmatched host TinyCNN budget",
                precision="float32 checkpoint evaluated in NumPy float64",
                evidence_level="project checkpoint diagnostic replay",
            ),
            _budget(legacy_macs, 0, legacy_workspace, diagnostic_wall_clock, "float32 weights / float64 NumPy execution", "one residual update per 32-cycle window", "five windows with edge-repeat padding", provenance="live artifact shape accounting and T6.17.3 host diagnostic"),
            replay_state="DIAGNOSTIC_REPLAY_EXACT_NOT_RANKED",
        ),
        _candidate(
            "legacy_static_theta_tinycnn", "legacy_cnn_noise_parameter_estimator", "single_mode_decoder", STATIC_THETA_MODEL, 1,
            _signature(decision_target="static noise-parameter regression", input_semantics="single 32x32 histogram", history_horizon="one aggregated histogram", output_action="sigma/mu_q/mu_p/theta estimates", noise_model="static anisotropic Gaussian", time_basis="one estimate per dataset sample", compute_budget="unmatched host TinyCNN budget", precision="float32 checkpoint / NumPy float64", evidence_level="project checkpoint not replayed because wrong action"),
            _budget(12 * 32 * 32 * 3 * 3 + 3072 * 64 + 64 * 4, 0, 3 * 3 * 32 * 32 * 8, None, "float32 weights / float64 NumPy execution", "dataset sample", "none reported", provenance="live artifact shape accounting"),
        ),
        _candidate(
            "t411_causal_tcn", "causal_adaptive_nn_regime_estimator", "single_mode_decoder", T411_CHECKPOINT, 1,
            _signature(decision_target="four-regime posterior classification", input_semantics="14 observed summaries per non-overlapping window", history_horizon="eight 32-cycle windows", output_action="four-class regime posterior", time_basis="one update per 32-cycle window", compute_budget="T4.1.1 slow-loop budget", precision="torch float32", evidence_level="project checkpoint parent replay only"),
            _budget(profiles["causal_tcn"]["macs_per_update_proxy"], profiles["causal_tcn"]["model_and_state_bytes"], profiles["causal_tcn"]["transient_workspace_bytes"], {"value": profiles["causal_tcn"]["host_batch_median_us_per_update"], "unit": "us/update", "boundary": "T4.1.1 host batch diagnostic"}, "torch float32", "one update per 32-cycle window", "eight windows", provenance="T4.1.1 resource profile"),
        ),
        _candidate(
            "t411_small_gru", "causal_adaptive_nn_regime_estimator", "single_mode_decoder", T411_CHECKPOINT, 1,
            _signature(decision_target="four-regime posterior classification", input_semantics="14 observed summaries per non-overlapping window", history_horizon="eight 32-cycle windows", output_action="four-class regime posterior", time_basis="one update per 32-cycle window", compute_budget="T4.1.1 slow-loop budget", precision="torch float32", evidence_level="project checkpoint parent replay only"),
            _budget(profiles["small_gru"]["macs_per_update_proxy"], profiles["small_gru"]["model_and_state_bytes"], profiles["small_gru"]["transient_workspace_bytes"], {"value": profiles["small_gru"]["host_batch_median_us_per_update"], "unit": "us/update", "boundary": "T4.1.1 host batch diagnostic"}, "torch float32", "one update per 32-cycle window", "eight windows", provenance="T4.1.1 resource profile"),
        ),
        _candidate(
            "t327_latest_outcome_fnn", "latest_outcome_neural_controller", "controller_rl_nmf", T327_CHECKPOINT, 5,
            controller_signature | {
                "input_semantics": "current observed g/e/leakage token plus static protocol features",
                "history_horizon": "strict latest outcome only",
                "compute_budget": "T3.2.7 matched history-controller parameter/MAC budget",
                "precision": "torch float64",
                "evidence_level": "project controller checkpoint family",
            },
            _budget(
                t327["compute_contract"]["total_dense_macs"],
                0,
                None,
                None,
                "torch float64",
                "one action per half-cycle",
                "no recurrent state; current observed outcome required",
                provenance="T3.2.7 compute contract and all-seed checkpoint",
            ),
        ),
        _candidate(
            "t3210_exponential_recurrence", "causal_adaptive_control_student", "controller_rl_nmf", T3210_CHECKPOINT, 3,
            controller_signature | {
                "input_semantics": "observed g/e/leakage outcome token",
                "history_horizon": "15-state exponential recurrence",
                "compute_budget": "T3.2.10 O(15) recurrence budget",
                "precision": "torch float64",
                "evidence_level": "project controller checkpoint family",
            },
            _budget(
                15,
                t3210["recurrence_contract"]["online_state_scalars"] * 8,
                None,
                None,
                "torch float64",
                "one action per half-cycle",
                "registered initial 15-control state",
                provenance="T3.2.10 recurrence contract and all-restart checkpoint",
            ),
        ),
        _candidate(
            "t415_distilled_recurrence_student", "causal_adaptive_control_student", "controller_rl_nmf", T415_ARTIFACT, 3,
            controller_signature | {
                "input_semantics": "observed outcome and six health/validity fields",
                "history_horizon": "15-state exponential recurrence",
                "compute_budget": "T4.1.5 deterministic online student budget",
                "precision": "float64 JSON artifact",
                "evidence_level": "project controller artifact",
            },
            _budget(
                t415["student_artifact"]["resource_profile"]["multiplications_per_healthy_step"],
                t415["student_artifact"]["resource_profile"]["state_scalars"] * 8,
                None,
                None,
                "float64 JSON artifact",
                "one action per half-cycle",
                "registered 15-control initial state",
                provenance="T4.1.5 selected artifact and online resource profile",
            ),
        ),
        _candidate(
            "t441_bounded_residual_gru_teacher", "offline_teacher", "controller_rl_nmf", T441_CHECKPOINT, 3,
            controller_signature,
            _budget(72_266, 10 * 8, None, None, "torch float64", "one action per half-cycle", "recurrent zero state / protocol reset", provenance="T4.4.1/T4.4.4 parent accounting"),
        ),
        _candidate(
            "t443_distilled_state4_student", "causal_adaptive_control_student", "controller_rl_nmf", T443_ARTIFACT, 9,
            controller_signature | {"input_semantics": "observed binary outcome and health flags", "history_horizon": "four-state exponential recurrence", "compute_budget": "T4.4.3/T5.5.4 controller budget", "precision": "float64 artifact and Q3.14 RTL side evidence", "evidence_level": "project controller artifact"},
            _budget(t443["resource_profile"]["multiply_adds_per_healthy_step"], t443["resource_profile"]["persistent_state_scalars"] * 8, None, {"value": student["worst_case_latency_us_at_27mhz"], "unit": "us/update", "boundary": "CXXRTL/P&R estimate, not board"}, "float64 artifact / Q3.14 RTL", "one action per half-cycle", "four-state reset vector", provenance="T4.4.3 artifact and T5.5.4 report"),
        ),
        _candidate(
            "t545_horizon_student_family", "causal_adaptive_control_student", "controller_rl_nmf", T545_CHECKPOINT, 10,
            controller_signature | {"input_semantics": "observed binary outcome stream", "history_horizon": "2/5/10/32-cycle trained recurrent candidates", "compute_budget": "T5.4.5 diagnostic horizon budget", "precision": "torch float64/float32 stability replay", "evidence_level": "project controller checkpoint family"},
            _budget(87, 4 * 8, None, None, "torch float64 with float32 stability replay", "one action per half-cycle", "horizon-specific recurrent state", provenance="T5.4.5 parent checkpoint/report"),
        ),
        _candidate(
            "t554_quantized_gru_shadow", "quantized_offline_teacher_shadow", "controller_rl_nmf", T554_MANIFEST, 1,
            controller_signature | {"compute_budget": "72,854-cycle optimistic lower-bound workload", "precision": "int8 weights/Q3.14 state functional shadow", "evidence_level": "project nonfunctional RTL workload"},
            _budget(quantized["analytic_weight_macs_per_update"], 10 * 4, None, {"value": quantized["latency_us_at_27mhz_lower_bound"], "unit": "us/update lower bound", "boundary": "nonfunctional workload"}, "int8/Q3.14", "one action per half-cycle", "GRU hidden reset", provenance="T5.5.4 report/manifest"),
        ),
        _candidate(
            "t237_project_nmf_controller", "model_based_nmf_controller", "controller_rl_nmf", T237_CHECKPOINT, 5,
            controller_signature | {"input_semantics": "binary ancilla history", "history_horizon": "recurrent controller history", "precision": "torch float64", "evidence_level": "project controller checkpoint"},
            _budget(72_266, 10 * 8, None, None, "torch float64", "one action per half-cycle", "recurrent state reset", provenance="T2.3.7 checkpoint/T4.4.4 cost table"),
            source_id="PUVIANI2025_NMF",
        ),
        _candidate(
            "gqf_official_nmf_controller", "official_source_controller", "controller_rl_nmf", GQF_INTAKE, 1,
            controller_signature | {"evidence_level": "official intake without usable paper checkpoint"},
            _budget(None, None, None, None, None, "source protocol", "source protocol", provenance="T6.8.3 official intake; exact paper reproduction remained blocked"),
            source_id="PUVIANI2025_NMF",
        ),
        _candidate(
            "wang2022_direct_nn", "external_direct_nn", source_rows["WANG2022_DIRECT_NN"]["lane_id"], None, 1,
            _signature(code_family="surface-GKP multidimensional construction", modes_or_distance="source finite-size surface-GKP family", decision_target=source_rows["WANG2022_DIRECT_NN"]["decision_object"], input_semantics=source_rows["WANG2022_DIRECT_NN"]["input_history"], history_horizon="source-defined syndrome features", output_action=source_rows["WANG2022_DIRECT_NN"]["output_action"], noise_model=source_rows["WANG2022_DIRECT_NN"]["noise_model"], time_basis="source decode instance", compute_budget="not reported", precision="not reported", evidence_level="literature only; no exact public checkpoint"),
            _budget(None, None, None, None, None, None, None, provenance="WANG2022 source record"),
            source_id="WANG2022_DIRECT_NN",
        ),
        _candidate(
            "sivak2023_rl_controller", "external_rl_controller", source_rows["SIVAK2023_RL_GKP"]["lane_id"], None, 1,
            controller_signature | {"code_family": "experimental cavity-transmon GKP", "input_semantics": source_rows["SIVAK2023_RL_GKP"]["input_history"], "history_horizon": "300 episodes x 10 candidates per epoch", "output_action": source_rows["SIVAK2023_RL_GKP"]["output_action"], "noise_model": source_rows["SIVAK2023_RL_GKP"]["noise_model"], "time_basis": "optimization epoch / fixed deployed circuit", "evidence_level": "literature only"},
            _budget(None, None, None, {"value": 16.0, "unit": "s/optimization epoch", "boundary": "not inference latency"}, None, "offline experiment-in-loop epoch", "not reported", provenance="SIVAK2023 source record"),
            source_id="SIVAK2023_RL_GKP",
        ),
        _candidate(
            "sivak2026_rl_drift", "external_rl_controller", source_rows["SIVAK2026_RL_DRIFT"]["lane_id"], None, 1,
            controller_signature | {"code_family": "surface/color-code processor", "modes_or_distance": "d=5/d=7 source experiments", "decision_target": source_rows["SIVAK2026_RL_DRIFT"]["decision_object"], "input_semantics": source_rows["SIVAK2026_RL_DRIFT"]["input_history"], "history_horizon": "source experiment-in-loop history", "output_action": source_rows["SIVAK2026_RL_DRIFT"]["output_action"], "noise_model": source_rows["SIVAK2026_RL_DRIFT"]["noise_model"], "time_basis": "adaptation epochs", "evidence_level": "literature only"},
            _budget(None, None, None, None, None, "adaptation epoch", "source-specific", provenance="SIVAK2026 source record"),
            source_id="SIVAK2026_RL_DRIFT",
        ),
    ]
    return rows


def _legacy_replay() -> dict[str, Any]:
    parent = _load(T543)["lanes"]["cnn_residual"]
    preserved = np.asarray([row["active_predicted_delta_b"] for row in parent["samples"]], dtype=np.float64)
    with np.load(LEGACY_INPUT, allow_pickle=False) as data:
        histograms = data["histograms"].astype(np.float64)
        labels = data["labels"].astype(np.float64)
        scenarios = data["scenario_names"].astype(str)
        window_ids = data["window_ids"].astype(np.int64)
    output_hashes: list[str] = []
    predictions: list[NDArray[np.float64]] = []
    runtimes: list[float] = []
    for _ in range(5):
        start = perf_counter()
        output = np.asarray(predict_from_artifact(LEGACY_MODEL, histograms), dtype=np.float64)
        runtimes.append(perf_counter() - start)
        output_hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
        predictions.append(output)
    prediction = predictions[0]
    squared = np.mean((prediction - labels) ** 2, axis=1)
    rows = [
        {
            "sample_id": index,
            "scenario_id": str(scenarios[index]),
            "window_id": int(window_ids[index]),
            "target": labels[index].tolist(),
            "prediction": prediction[index].tolist(),
            "squared_error": float(squared[index]),
        }
        for index in range(prediction.shape[0])
    ]
    return {
        "state": "DIAGNOSTIC_REPLAY_EXACT_NOT_RANKED",
        "reason_not_ranked": "input is a five-window histogram/teacher-feature tensor and output is b_q/b_p regression, not the frozen per-round syndrome-to-logical-action task",
        "samples": int(prediction.shape[0]),
        "label_names": ["b_q", "b_p"],
        "model_sha256": _sha256(LEGACY_MODEL),
        "input_file_sha256": _sha256(LEGACY_INPUT),
        "input_array_sha256": hashlib.sha256(histograms.tobytes()).hexdigest(),
        "output_sha256": output_hashes[0],
        "repeat_output_sha256s": output_hashes,
        "repeat_count": len(output_hashes),
        "bit_exact_across_repeats": len(set(output_hashes)) == 1,
        "bit_exact_with_t5_4_3_preserved_predictions": bool(np.array_equal(prediction, preserved)),
        "maximum_abs_difference_from_t5_4_3": float(np.max(np.abs(prediction - preserved))),
        "mse": float(np.mean(squared)),
        "mae": float(np.mean(np.abs(prediction - labels))),
        "parent_mse": parent["aggregate"]["active_mse"],
        "parent_evaluation_report_mse": parent["aggregate"]["preserved_evaluation_report_mse"],
        "host_batch_samples": int(prediction.shape[0]),
        "host_batch_runtime_seconds": runtimes,
        "host_batch_median_seconds": float(np.median(runtimes)),
        "host_timing_boundary": "whole NumPy batch on current host; diagnostic only, not decoder latency_ns",
        "rows": rows,
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in report["candidates"]:
        rows.append({"record_type": "candidate", "record_id": candidate["candidate_id"], "candidate_id": candidate["candidate_id"], "field": "eligibility", "value": candidate["eligibility_state"], "value_state": "INELIGIBLE", "eligible": candidate["same_task_eligible"], "details": json.dumps(candidate["mismatch_fields"], sort_keys=True)})
        for field, check in candidate["signature_checks"].items():
            rows.append({"record_type": "signature", "record_id": f"{candidate['candidate_id']}:{field}", "candidate_id": candidate["candidate_id"], "field": field, "value": check["actual"], "value_state": "MATCH" if check["match"] else "MISMATCH", "eligible": check["match"], "details": f"required={check['required']}"})
    for row in report["diagnostic_replay"]["rows"]:
        rows.append({"record_type": "diagnostic_replay", "record_id": str(row["sample_id"]), "candidate_id": "legacy_residual_tinycnn", "field": "b_q/b_p_prediction", "value": json.dumps(row["prediction"]), "value_state": "DIAGNOSTIC_NOT_RANKED", "eligible": False, "details": json.dumps({"scenario_id": row["scenario_id"], "window_id": row["window_id"], "target": row["target"], "squared_error": row["squared_error"]}, sort_keys=True)})
    for source_id, source_hash in report["source_scope"].items():
        rows.append({"record_type": "source", "record_id": source_id, "candidate_id": "", "field": "source_record_sha256", "value": source_hash, "value_state": "LITERATURE_ONLY", "eligible": False, "details": "source-scoped; no cross-lane replay"})
    return rows


def _write_csv(report: Mapping[str, Any]) -> None:
    fields = ["record_type", "record_id", "candidate_id", "field", "value", "value_state", "eligible", "details"]
    with DEFAULT_SOURCE_DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_source_rows(report))


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    frozen = _preregistered_experiment()
    candidates = report["candidates"]
    replay = report["diagnostic_replay"]
    bindings = report["bindings"]
    source_ids = {"WANG2022_DIRECT_NN", "SIVAK2023_RL_GKP", "SIVAK2026_RL_DRIFT", "PUVIANI2025_NMF"}
    expected_candidate_ids = {
        "legacy_residual_tinycnn", "legacy_static_theta_tinycnn", "t411_causal_tcn", "t411_small_gru",
        "t327_latest_outcome_fnn", "t3210_exponential_recurrence", "t415_distilled_recurrence_student",
        "t441_bounded_residual_gru_teacher", "t443_distilled_state4_student", "t545_horizon_student_family",
        "t554_quantized_gru_shadow", "t237_project_nmf_controller", "gqf_official_nmf_controller",
        "wang2022_direct_nn", "sivak2023_rl_controller", "sivak2026_rl_drift",
    }
    return {
        "G01_frozen_readonly_preregistration_is_consumed_exactly": report["preregistration"]["record_sha256"] == _canonical_sha256(frozen) and frozen["config"]["training_allowed"] is False and frozen["config"]["hyperparameter_search_allowed"] is False and frozen["config"]["checkpoint_reselection_allowed"] is False,
        "G02_candidate_universe_covers_all_relevant_existing_families_without_temp_artifacts": len(candidates) == len(expected_candidate_ids) == 16 and {row["candidate_id"] for row in candidates} == expected_candidate_ids and all(row["member_count"] >= 1 and (row["artifact"] is None or not row["artifact"]["path"].startswith(("tmp/", ".pytest"))) for row in candidates),
        "G03_direct_nn_causal_nn_rl_and_nmf_are_distinct_categories_and_lanes": {row["category"] for row in candidates} >= {"external_direct_nn", "causal_adaptive_nn_regime_estimator", "external_rl_controller", "model_based_nmf_controller"} and next(row for row in candidates if row["candidate_id"] == "wang2022_direct_nn")["native_lane"] == "surface_gkp_gate_outer_code" and all(next(row for row in candidates if row["candidate_id"] == item)["native_lane"] == "controller_rl_nmf" for item in ("sivak2023_rl_controller", "sivak2026_rl_drift", "t237_project_nmf_controller", "gqf_official_nmf_controller")),
        "G04_every_candidate_has_complete_13_field_signature_and_live_mismatch_projection": all(
            set(row["signature"]) == set(SIGNATURE_FIELDS)
            and set(row["signature_checks"]) == set(SIGNATURE_FIELDS)
            and row["signature_checks"]
            == {
                field: {
                    "required": REQUIRED_SIGNATURE[field],
                    "actual": row["signature"][field],
                    "match": row["signature"][field] == REQUIRED_SIGNATURE[field],
                }
                for field in SIGNATURE_FIELDS
            }
            and row["mismatch_fields"]
            == [field for field in SIGNATURE_FIELDS if row["signature"][field] != REQUIRED_SIGNATURE[field]]
            and row["same_task_eligible"] == (len(row["mismatch_fields"]) == 0)
            for row in candidates
        ),
        "G05_every_candidate_has_all_seven_frozen_budget_fields_even_when_null": frozen["config"]["required_budget_fields"] == REQUIRED_BUDGET_FIELDS and all(set(row["budget"]) == set(REQUIRED_BUDGET_FIELDS) | {"provenance"} for row in candidates),
        "G06_no_existing_checkpoint_matches_same_syndrome_action_precision_and_budget": report["eligibility_summary"] == {"candidate_families": 16, "same_task_eligible": 0, "eligible_replayed": 0, "ineligible": 16, "diagnostic_replays": 1} and all(not row["same_task_eligible"] and row["mismatch_fields"] for row in candidates),
        "G07_ineligible_rows_keep_all_primary_metrics_null_and_unranked": all(set(row["metrics"]) == set(METRICS) and all(metric["value"] is None and metric["value_state"] == "N_A_NOT_APPLICABLE" and not metric["ranking_eligible"] for metric in row["metrics"].values()) for row in candidates),
        "G08_legacy_cnn_readonly_reinference_is_bit_exact_with_parent_and_repeats": replay["state"] == "DIAGNOSTIC_REPLAY_EXACT_NOT_RANKED" and replay["samples"] == 206 and replay["repeat_count"] == 5 and replay["bit_exact_across_repeats"] and replay["bit_exact_with_t5_4_3_preserved_predictions"] and replay["maximum_abs_difference_from_t5_4_3"] == 0.0,
        "G09_legacy_cnn_diagnostic_metrics_recompute_parent_without_ler_promotion": np.isclose(replay["mse"], np.mean([row["squared_error"] for row in replay["rows"]]), rtol=0.0, atol=1e-20) and np.isclose(replay["mse"], replay["parent_mse"], rtol=0.0, atol=1e-20) and np.isclose(replay["mse"], replay["parent_evaluation_report_mse"], rtol=0.0, atol=1e-20) and "not decoder latency_ns" in replay["host_timing_boundary"],
        "G10_checkpoint_input_output_and_parent_hashes_are_complete": replay["model_sha256"] == _sha256(LEGACY_MODEL) and replay["input_file_sha256"] == _sha256(LEGACY_INPUT) and len(replay["input_array_sha256"]) == len(replay["output_sha256"]) == 64 and len(set(replay["repeat_output_sha256s"])) == 1 and replay["output_sha256"] == replay["repeat_output_sha256s"][0] and next(row for row in candidates if row["candidate_id"] == "legacy_residual_tinycnn")["artifact"]["sha256"] == replay["model_sha256"],
        "G11_external_methods_without_exact_checkpoint_remain_literature_only_and_unreplayed": set(report["source_scope"]) == source_ids and all(next(row for row in candidates if row["candidate_id"] == candidate_id)["artifact"] is None and next(row for row in candidates if row["candidate_id"] == candidate_id)["replay_state"] == "NOT_REPLAYED_INELIGIBLE" for candidate_id in ("wang2022_direct_nn", "sivak2023_rl_controller", "sivak2026_rl_drift")) and next(row for row in candidates if row["candidate_id"] == "gqf_official_nmf_controller")["replay_state"] == "NOT_REPLAYED_INELIGIBLE",
        "G12_controller_outputs_are_never_merged_into_direct_decoder_rank": all("output_action" in row["mismatch_fields"] for row in candidates if row["native_lane"] == "controller_rl_nmf") and report["cross_lane_aggregate"] is None and report["global_ranking"] is None,
        "G13_no_training_reselection_or_phase6b_claim_upgrade_occurred": report["execution_contract"] == {"training_executed": False, "hyperparameter_search_executed": False, "checkpoint_reselection_executed": False, "new_checkpoint_written": False, "performance_p_value_computed": False, "phase6b_outputs_modified": False} and report["claim_registry"]["PHASE6B_V5_VERDICT"] == "READ_ONLY_NO_GO_UNCHANGED" and _load(T6155)["verdict"] == "NO_GO_V5_EARLY_HEADROOM_STOP",
        "G14_runtime_budget_is_respected_and_host_batch_time_is_not_latency": report["execution_budget_audit"]["within_runtime_budget"] and report["execution_budget_audit"]["within_memory_budget"] and replay["host_batch_median_seconds"] > 0.0 and all(metric["value"] is None for row in candidates for name, metric in row["metrics"].items() if name == "latency_ns"),
        "G15_source_data_semantic_locks_and_exact_bindings_are_live": report["source_data"]["rows"] == 16 + 16 * 13 + 206 + 4 and report["source_data"]["sha256"] == _sha256(ROOT / report["source_data"]["path"]) and bindings["source_data"]["sha256"] == report["source_data"]["sha256"] and all(_sha256(ROOT / row["path"]) == row["sha256"] for name, row in bindings.items() if name not in {"ontology_initial", "source_audit_initial"}) and _canonical_sha256(_ontology_semantic(_load(ONTOLOGY))) == report["ontology_semantic_sha256"] and _canonical_sha256(_source_semantic(_load(SOURCE_AUDIT))) == report["source_audit_semantic_sha256"],
        "G16_targeted_semantic_mutations_are_all_detected": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 16 and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]),
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("enable_posthoc_training", "G01_frozen_readonly_preregistration_is_consumed_exactly", lambda x: x["preregistration"].update(record_sha256="0" * 64))
    attempt("drop_candidate_family", "G02_candidate_universe_covers_all_relevant_existing_families_without_temp_artifacts", lambda x: x["candidates"].pop())
    attempt("merge_wang_with_rl", "G03_direct_nn_causal_nn_rl_and_nmf_are_distinct_categories_and_lanes", lambda x: next(row for row in x["candidates"] if row["candidate_id"] == "wang2022_direct_nn").update(native_lane="controller_rl_nmf"))
    attempt("forge_signature_match", "G04_every_candidate_has_complete_13_field_signature_and_live_mismatch_projection", lambda x: next(row for row in x["candidates"] if row["candidate_id"] == "legacy_residual_tinycnn")["signature_checks"]["output_action"].update(match=True))
    attempt("remove_workspace_budget", "G05_every_candidate_has_all_seven_frozen_budget_fields_even_when_null", lambda x: next(row for row in x["candidates"] if row["candidate_id"] == "wang2022_direct_nn")["budget"].pop("workspace_bytes"))
    attempt("promote_ineligible_checkpoint", "G06_no_existing_checkpoint_matches_same_syndrome_action_precision_and_budget", lambda x: x["eligibility_summary"].update(same_task_eligible=1))
    attempt("fill_ineligible_ler", "G07_ineligible_rows_keep_all_primary_metrics_null_and_unranked", lambda x: next(row for row in x["candidates"] if row["candidate_id"] == "legacy_residual_tinycnn")["metrics"]["p_L"].update(value=0.01))
    attempt("break_replay_exactness", "G08_legacy_cnn_readonly_reinference_is_bit_exact_with_parent_and_repeats", lambda x: x["diagnostic_replay"].update(bit_exact_with_t5_4_3_preserved_predictions=False))
    attempt("rename_parameter_mse_as_ler", "G09_legacy_cnn_diagnostic_metrics_recompute_parent_without_ler_promotion", lambda x: x["diagnostic_replay"].update(host_timing_boundary="decoder latency_ns"))
    attempt("forge_output_hash", "G10_checkpoint_input_output_and_parent_hashes_are_complete", lambda x: x["diagnostic_replay"].update(output_sha256="0" * 64))
    attempt("claim_external_checkpoint_replayed", "G11_external_methods_without_exact_checkpoint_remain_literature_only_and_unreplayed", lambda x: next(row for row in x["candidates"] if row["candidate_id"] == "wang2022_direct_nn").update(replay_state="PROJECT_NATIVE_MATCHED"))
    attempt("create_cross_lane_score", "G12_controller_outputs_are_never_merged_into_direct_decoder_rank", lambda x: x.update(global_ranking=["winner"]))
    attempt("claim_new_training", "G13_no_training_reselection_or_phase6b_claim_upgrade_occurred", lambda x: x["execution_contract"].update(training_executed=True))
    attempt("invent_latency", "G14_runtime_budget_is_respected_and_host_batch_time_is_not_latency", lambda x: next(row for row in x["candidates"] if row["candidate_id"] == "legacy_residual_tinycnn")["metrics"]["latency_ns"].update(value=50.0))
    attempt("forge_source_hash", "G15_source_data_semantic_locks_and_exact_bindings_are_live", lambda x: x["source_data"].update(sha256="0" * 64))
    attempt("forge_mutation_count", "G16_targeted_semantic_mutations_are_all_detected", lambda x: x.update(semantic_mutation_audit={"count": 16, "detected": 15, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    frozen = _preregistered_experiment()
    start = perf_counter()
    replay = _legacy_replay()
    diagnostic_wall_clock = {
        "value": replay["host_batch_median_seconds"],
        "unit": "s/206-sample batch",
        "boundary": "current-host NumPy diagnostic; not decoder latency",
    }
    candidates = _candidate_registry(diagnostic_wall_clock)
    source_scope = {
        source_id: _canonical_sha256(next(row for row in _load(SOURCE_AUDIT)["sources"] if row["source_id"] == source_id))
        for source_id in ("WANG2022_DIRECT_NN", "SIVAK2023_RL_GKP", "SIVAK2026_RL_DRIFT", "PUVIANI2025_NMF")
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only checkpoint eligibility and diagnostic replay; no post-T6.15.5 training, selection or V5 promotion",
        "preregistration": {
            "experiment_id": frozen["experiment_id"],
            "record_sha256": _canonical_sha256(frozen),
            "split": frozen["split"],
            "required_budget_fields": frozen["config"]["required_budget_fields"],
        },
        "universe_contract": {
            "included_parents": ["T3.2.7", "T3.2.10", "T4.1.1", "T4.1.5", "T4.4.1", "T4.4.3", "T5.4.3", "T5.4.5", "T5.5.4", "T2.3.7", "T6.8.3", "T6.16.1"],
            "grouping_rule": "restarts/candidate horizons with identical decision object are one family row; member_count preserves multiplicity",
            "excluded": ["pytest/temp checkpoints", "training-only scratch files", "classical HMM/Kalman/FSM rows", "newly trained or reselected models"],
            "selection_rule": "no performance selection; eligibility depends only on frozen task signature and budget",
        },
        "required_signature": REQUIRED_SIGNATURE,
        "signature_fields": SIGNATURE_FIELDS,
        "candidates": candidates,
        "eligibility_summary": {
            "candidate_families": len(candidates),
            "same_task_eligible": sum(row["same_task_eligible"] for row in candidates),
            "eligible_replayed": 0,
            "ineligible": sum(not row["same_task_eligible"] for row in candidates),
            "diagnostic_replays": 1,
        },
        "diagnostic_replay": replay,
        "source_scope": source_scope,
        "execution_contract": {
            "training_executed": False,
            "hyperparameter_search_executed": False,
            "checkpoint_reselection_executed": False,
            "new_checkpoint_written": False,
            "performance_p_value_computed": False,
            "phase6b_outputs_modified": False,
        },
        "cross_lane_aggregate": None,
        "global_ranking": None,
        "claim_registry": {
            "SAME_TASK_LEARNED_DECODER_PERFORMANCE": "NOT_ESTABLISHED_NO_ELIGIBLE_CHECKPOINT",
            "LEGACY_CNN_PARAMETER_REPLAY": "DIAGNOSTIC_EXACT_INELIGIBLE",
            "WANG_DIRECT_NN": "LITERATURE_ONLY_DIFFERENT_SURFACE_GKP_LANE",
            "SIVAK_RL": "LITERATURE_ONLY_CONTROLLER_LANE",
            "PUVIANI_NMF": "CONTROLLER_LANE_NOT_DIRECT_DECODER",
            "CNN_OR_RL_LATENCY": "NULL_UNMATCHED_BOUNDARY",
            "PHASE6B_V5_VERDICT": "READ_ONLY_NO_GO_UNCHANGED",
        },
        "allowed_wording": [
            "The preserved legacy TinyCNN can be re-inferred bit-exactly on its 206-sample residual-parameter test split, but it is not a same-task logical decoder.",
            "No existing learned checkpoint matches the frozen single-mode syndrome/action/precision/compute signature; learned methods remain diagnostic or separate controller/surface-GKP evidence.",
        ],
        "forbidden_wording": [
            "The legacy CNN has a logical-error or latency advantage over the project decoders.",
            "Wang Direct NN, Sivak RL and Puviani NMF form one decoder leaderboard.",
            "A controller/student checkpoint is a per-round Pauli decoder because it is causal or neural.",
        ],
    }
    elapsed = perf_counter() - start
    report["execution_budget_audit"] = {
        "runtime_seconds": elapsed,
        "runtime_budget_seconds": frozen["runtime_budget"]["wall_clock_seconds"],
        "memory_budget_bytes": int(frozen["runtime_budget"]["memory_gib"] * (1 << 30)),
        "estimated_peak_bytes_upper_bound": LEGACY_INPUT.stat().st_size + 21 * 3 * 3 * 32 * 32 * 8 + LEGACY_MODEL.stat().st_size,
        "within_runtime_budget": elapsed <= frozen["runtime_budget"]["wall_clock_seconds"],
        "within_memory_budget": LEGACY_INPUT.stat().st_size + 21 * 3 * 3 * 32 * 32 * 8 + LEGACY_MODEL.stat().st_size <= frozen["runtime_budget"]["memory_gib"] * (1 << 30),
        "boundary": "eligibility audit plus one diagnostic NumPy batch; not a decoder latency benchmark",
    }
    _write_csv(report)
    report["source_data"] = {"path": _relative(DEFAULT_SOURCE_DATA), "sha256": _sha256(DEFAULT_SOURCE_DATA), "rows": sum(1 for _ in DEFAULT_SOURCE_DATA.open(encoding="utf-8")) - 1}
    report["ontology_semantic_sha256"] = _canonical_sha256(_ontology_semantic(_load(ONTOLOGY)))
    report["source_audit_semantic_sha256"] = _canonical_sha256(_source_semantic(_load(SOURCE_AUDIT)))
    report["bindings"] = {
        "implementation": _binding(Path(__file__)),
        "preregistration_config": _binding(PREREG_CONFIG),
        "ontology_initial": _binding(ONTOLOGY),
        "source_audit_initial": _binding(SOURCE_AUDIT),
        "phase6b_terminal": _binding(T6155),
        "algorithm_verdict": _binding(T514),
        "causal_ablation_parent": _binding(T543),
        "t327_report": _binding(T327_REPORT),
        "t327_checkpoint": _binding(T327_CHECKPOINT),
        "t3210_report": _binding(T3210_REPORT),
        "t3210_checkpoint": _binding(T3210_CHECKPOINT),
        "legacy_manifest": _binding(LEGACY_MANIFEST),
        "legacy_input": _binding(LEGACY_INPUT),
        "legacy_model": _binding(LEGACY_MODEL),
        "legacy_eval": _binding(LEGACY_EVAL),
        "t411_report": _binding(T411_REPORT),
        "t411_checkpoint": _binding(T411_CHECKPOINT),
        "t415_report": _binding(T415_REPORT),
        "t415_artifact": _binding(T415_ARTIFACT),
        "t441_checkpoint": _binding(T441_CHECKPOINT),
        "t443_artifact": _binding(T443_ARTIFACT),
        "t443_checkpoint": _binding(T443_CHECKPOINT),
        "t545_checkpoint": _binding(T545_CHECKPOINT),
        "t554_report": _binding(T554_REPORT),
        "t554_manifest": _binding(T554_MANIFEST),
        "t237_checkpoint": _binding(T237_CHECKPOINT),
        "gqf_intake": _binding(GQF_INTAKE),
        "source_data": _binding(DEFAULT_SOURCE_DATA),
    }
    report["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    failed = [name for name, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {"passed": len(report["gates"]) - len(failed), "failed": failed}
    report["verdict"] = VERDICT if not failed else "FAIL_LEARNED_MODEL_ELIGIBILITY_AUDIT"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if dict(report["gates"]) != gates:
        raise ValueError("stored T6.17.3 gates do not match recomputation")
    failed = [name for name, passed in gates.items() if not passed]
    expected_summary = {"passed": len(gates) - len(failed), "failed": failed}
    expected_verdict = VERDICT if not failed else "FAIL_LEARNED_MODEL_ELIGIBILITY_AUDIT"
    if report["gate_summary"] != expected_summary or report["verdict"] != expected_verdict:
        raise ValueError("stored T6.17.3 summary/verdict does not match recomputation")
    if report["source_data"]["sha256"] != _sha256(ROOT / report["source_data"]["path"]):
        raise ValueError("T6.17.3 Source Data drifted")


def write_markdown(report: Mapping[str, Any], path: Path = DEFAULT_MARKDOWN) -> None:
    replay = report["diagnostic_replay"]
    lines = [
        "# T6.17.3 learned model eligibility 与只读 replay",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- candidate families：{report['eligibility_summary']['candidate_families']}；same-task eligible={report['eligibility_summary']['same_task_eligible']}；diagnostic replay={report['eligibility_summary']['diagnostic_replays']}",
        f"- legacy CNN replay：{replay['samples']} samples，MSE={replay['mse']:.9g}，parent max-abs diff={replay['maximum_abs_difference_from_t5_4_3']:.3g}",
        f"- gates / mutations：{report['gate_summary']['passed']}/16 / {report['semantic_mutation_audit']['detected']}/16；Source Data={report['source_data']['rows']} rows",
        "",
        "## Eligibility 结论",
        "",
        "| candidate | category | native lane | mismatched signature fields | replay |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["candidates"]:
        lines.append(f"| `{row['candidate_id']}` | `{row['category']}` | `{row['native_lane']}` | {', '.join(row['mismatch_fields'])} | `{row['replay_state']}` |")
    lines += [
        "",
        "没有 checkpoint 同时匹配 syndrome/action、observed-only history、cadence/warm-up、production fixed-point、MAC/state/workspace/wall-clock budget 和 parent trace。因而本 task 不产生 learned-decoder `p_L/p_X/p_Y/p_Z/average_ler/latency_ns` 排名；这些字段对全部 ineligible rows 都是 null。",
        "",
        "## Legacy CNN diagnostic replay",
        "",
        f"保留模型 `{_relative(LEGACY_MODEL)}` 在冻结的 206-sample test split 上重推理 5 次，输出 hash 完全一致，并与 T5.4.3 保存的逐样本预测 bit-exact。该模型输入为 21-channel、5-window histograms 与 teacher 参数/差分，输出为连续 `b_q/b_p` residual；它证明 artifact 可重放，不证明 logical decoding、drift-control gain 或 latency advantage。host batch median={replay['host_batch_median_seconds']:.6g} s，也不转换为 `latency_ns`。",
        "",
        "## 方法边界",
        "",
        "Wang 2022 是 surface-GKP direct decoder，但无 exact public checkpoint 且 code/task 不同；Sivak 2023/2026 是 experiment-in-loop controller；Puviani NMF 与项目 teacher/student 输出 15 个物理控制参数。它们分别留在 surface-GKP 或 controller lane，不与 single-mode Pauli decoder 合并。T6.15.5 后没有训练、超参搜索、checkpoint 重选或新 checkpoint 写入。",
        "",
        "## 产物",
        "",
        f"- report：`{_relative(DEFAULT_REPORT)}`",
        f"- Source Data：`{report['source_data']['path']}`",
        f"- implementation：`{_relative(Path(__file__))}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_report(_load(DEFAULT_REPORT))
        print(f"verified {DEFAULT_REPORT}")
        return
    report = build_report()
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report)
    verify_report(report)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "candidates": len(report["candidates"]), "eligible": report["eligibility_summary"]["same_task_eligible"], "source_rows": report["source_data"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
