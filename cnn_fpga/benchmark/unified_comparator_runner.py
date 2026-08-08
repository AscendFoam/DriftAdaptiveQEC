"""T6.6.1 unified comparator adapters and matched-budget qualification runner.

This is an integration qualification, not the pre-registered T6.7 formal
benchmark.  It executes the real periodic Window/EWMA/Kalman implementations,
the frozen joint-MAP baseline, the privileged oracle, and the preserved legacy
CNN checkpoint.  Deployable algorithms receive only the T6.5.2 packet schema.

The phase-conditioned wire interface carries one scalar syndrome per fast
cycle, while the software joint decoder consumes a q/p pair.  The bridge below
therefore forms a logical comparison sample only from an adjacent phase-0 then
phase-1 packet pair.  The regime estimator remains packet-causal: on every fast
cycle it sees the newly decoded phase plus the last observed value of the other
phase.  No floating syndrome side channel is accepted by an adapter.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from cnn_fpga.benchmark.continuous_adaptive_map import _residuals_and_truth
from cnn_fpga.benchmark.mixed_scenario_matrix import MixedDecoderScenario
from cnn_fpga.benchmark.static_map_baseline import StaticMAPParameters
from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    scaled_periodic_kalman_config,
)
from cnn_fpga.decoder.regime_hmm import (
    GaussianRegimeHMM,
    REGIME_CLASSES,
    RegimeObservationWindow,
    summarize_regime_window,
)
from cnn_fpga.decoder.slow_loop_model_selection import RollingGaussianHMMAdapter
from cnn_fpga.model.tiny_cnn import predict_from_artifact
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTRuntime,
    software_decode_syndrome_code,
    software_encode_syndrome_for_replay,
)
from cnn_fpga.runtime.unified_execution_contract import (
    DEPLOYABLE_METHOD_IDS,
    OBSERVED_SCHEMA_ID,
    ContractViolation,
    ExecutionAccountingRecord,
    MatchedBudget,
    ObservedSyndromePacket,
    assert_accounting_conforms,
    canonical_method_manifests,
    contract_sha256,
    oracle_method_manifest,
    validate_observed_mapping_for_deployable,
    validate_wire_roundtrip,
)
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState
from physics.ideal_gkp_decoder import map_decode_2d
from physics.oracle_map import oracle_map_2d


TASK_ID = "T6.6.1"
SCHEMA_VERSION = "t6.6.1-unified-comparator-runner-v1"
PROTOCOL_ID = "ROUTE-A-COMPARATOR-INTEGRATION-QUALIFICATION-V1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t6_6_1_unified_comparator_runner.json")
DEFAULT_SOURCE_DATA = Path("docs/t6_6_1_unified_comparator_runner_source_data.csv")
HMM_CHECKPOINT = Path("artifacts/models/route_a/t4_1_1_gaussian_hmm.json")
CNN_MODEL = Path(
    "artifacts/models/runtime_b_residual_v1/"
    "tiny_cnn_20260401_083648_2fc740424c0d.npz"
)
CNN_MANIFEST = Path("artifacts/datasets/runtime_b_residual_v1/manifest.json")
CNN_TEST_SPLIT = Path("artifacts/datasets/runtime_b_residual_v1/test.npz")
CNN_PARENT_REPORT = Path("docs/t5_4_3_causal_ablation_negative_results.json")
PARENT_MIXED_REPORT = Path("docs/t5_1_2_mixed_scenario_matrix.json")

COMMON_TRACE_METHODS = (
    "standard_binning",
    "static_joint_map",
    "window_map",
    "ewma_adaptive_map",
    "kalman_adaptive_map",
    "proposed_route_a",
)
ABlation_ONLY_METHODS = ("legacy_cnn_residual",)


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _mean_nll_brier(posterior: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    probability = np.asarray(posterior, dtype=np.float64).reshape((-1, 4))
    target = np.asarray(truth, dtype=np.int64).reshape(-1)
    selected = np.clip(probability[np.arange(len(target)), target], 1.0e-300, 1.0)
    one_hot = np.eye(4, dtype=np.float64)[target]
    return (
        float(np.mean(-np.log(selected))),
        float(np.mean(np.sum(np.square(probability - one_hot), axis=1))),
    )


@dataclass(frozen=True)
class RunnerConfig:
    scenario_id: str = "calibration_shift"
    seed: int = 202607176601
    windows: int = 16
    calibration_windows: int = 4
    scalar_packets_per_parameter_window: int = 2048
    parameter_window_stride_cycles: int = 4000
    regime_window_cycles: int = 32
    prefix_windows: int = 8
    cnn_witness_samples: int = 4

    def __post_init__(self) -> None:
        if self.scenario_id not in ("static_gaussian", "calibration_shift", "burst_outlier"):
            raise ValueError("qualification scenario is not registered")
        for name in (
            "seed",
            "windows",
            "calibration_windows",
            "scalar_packets_per_parameter_window",
            "parameter_window_stride_cycles",
            "regime_window_cycles",
            "prefix_windows",
            "cnn_witness_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        budget = MatchedBudget()
        if self.windows < 16:
            raise ValueError("qualification needs at least sixteen windows")
        if not 0 < self.prefix_windows < self.windows:
            raise ValueError("prefix_windows must be a strict trace prefix")
        if self.scalar_packets_per_parameter_window != budget.parameter_window_samples:
            raise ValueError("parameter-window sample count drifted from T6.5.2")
        if self.parameter_window_stride_cycles != budget.parameter_update_period_cycles:
            raise ValueError("parameter-window stride drifted from T6.5.2")
        if self.regime_window_cycles != budget.regime_window_cycles:
            raise ValueError("regime cadence drifted from T6.5.2")
        if self.scalar_packets_per_parameter_window % 2:
            raise ValueError("q/p packet pairing requires an even scalar window")
        if self.scalar_packets_per_parameter_window > self.parameter_window_stride_cycles:
            raise ValueError("packet windows must not overlap")

    @property
    def pairs_per_parameter_window(self) -> int:
        return self.scalar_packets_per_parameter_window // 2


@dataclass(frozen=True)
class ObservedTraceWindow:
    window_id: int
    packets: tuple[ObservedSyndromePacket, ...]
    packet_sha256: str

    def __post_init__(self) -> None:
        if len(self.packets) < 2 or len(self.packets) % 2:
            raise ValueError("observed trace windows require complete q/p packet pairs")
        if not all(isinstance(item, ObservedSyndromePacket) for item in self.packets):
            raise TypeError("packets must contain ObservedSyndromePacket values")
        expected_hash = _canonical_sha256([asdict(item) for item in self.packets])
        if self.packet_sha256 != expected_hash:
            raise ValueError("observed window packet hash mismatch")


@dataclass(frozen=True)
class ObservedTrace:
    trace_id: str
    calibration_residuals: np.ndarray
    calibration_sha256: str
    windows: tuple[ObservedTraceWindow, ...]
    trace_sha256: str

    def __post_init__(self) -> None:
        calibration = np.asarray(self.calibration_residuals, dtype=np.float64)
        if calibration.ndim != 2 or calibration.shape[1] != 2:
            raise ValueError("calibration_residuals must have shape (samples,2)")
        if self.calibration_sha256 != _array_sha256(calibration, "<f8"):
            raise ValueError("calibration hash mismatch")
        object.__setattr__(self, "calibration_residuals", _readonly(calibration, np.float64))
        expected = _canonical_sha256(
            {
                "trace_id": self.trace_id,
                "calibration_sha256": self.calibration_sha256,
                "window_hashes": [item.packet_sha256 for item in self.windows],
            }
        )
        if self.trace_sha256 != expected:
            raise ValueError("observed trace hash mismatch")


@dataclass(frozen=True)
class TruthTraceWindow:
    window_id: int
    truth: np.ndarray
    state: DriftState
    truth_sha256: str

    def __post_init__(self) -> None:
        truth = np.asarray(self.truth, dtype=np.int64)
        if truth.ndim != 1 or np.any((truth < 0) | (truth > 3)):
            raise ValueError("truth must be a logical-class vector")
        if not isinstance(self.state, DriftState):
            raise TypeError("state must be DriftState")
        if self.truth_sha256 != _array_sha256(truth, "<i8"):
            raise ValueError("truth hash mismatch")
        object.__setattr__(self, "truth", _readonly(truth, np.int64))


@dataclass(frozen=True)
class IsolatedTruthTrace:
    trace_id: str
    windows: tuple[TruthTraceWindow, ...]


@dataclass(frozen=True)
class MethodRun:
    method_id: str
    trace_sha256: str
    decisions: tuple[np.ndarray, ...]
    posteriors: tuple[np.ndarray, ...]
    accounting: tuple[ExecutionAccountingRecord, ...]
    window_rows: tuple[dict[str, Any], ...]
    private_model_state_bytes: int
    max_update_macs: int
    max_transient_workspace_bytes: int
    actual_execution: str
    current_rtl_status: str
    ranking_status: str

    def __post_init__(self) -> None:
        if len(self.decisions) != len(self.posteriors) or not self.decisions:
            raise ValueError("method decisions and posteriors must be nonempty and aligned")
        for decisions, posterior in zip(self.decisions, self.posteriors, strict=True):
            decision_array = np.asarray(decisions, dtype=np.int64)
            posterior_array = np.asarray(posterior, dtype=np.float64).reshape((-1, 4))
            if decision_array.shape != (len(posterior_array),):
                raise ValueError("method decision/posterior shape mismatch")
            if np.any((decision_array < 0) | (decision_array > 3)):
                raise ValueError("logical decisions must lie in [0,3]")
            if not np.allclose(np.sum(posterior_array, axis=1), 1.0, atol=1.0e-10, rtol=0.0):
                raise ValueError("posterior rows must normalize")

    @property
    def decision_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in self.decisions:
            digest.update(np.asarray(value, dtype="<i8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class MethodCost:
    update_macs: int
    private_model_state_bytes: int
    transient_workspace_bytes: int
    derivation: str


def derive_method_costs() -> dict[str, MethodCost]:
    """Derive, rather than hand-wave, the method-private operation ledger.

    The periodic characteristic frontend is a shared exact 10-bit lookup/index
    transform verified separately.  Counts here cover method-private real MACs.
    Dense Kalman counts deliberately follow the NumPy execution path rather
    than assuming sparse-matrix hardware that has not been implemented.
    """

    state_dim = 10
    observation_dim = 5
    moment_postprocess = 4 * 32
    prediction = state_dim * state_dim + 2 * state_dim**3
    innovation = (
        observation_dim * state_dim
        + observation_dim * state_dim**2
        + observation_dim**2 * state_dim
        + state_dim**2 * observation_dim
    )
    solve = observation_dim**3 // 3 + observation_dim**2 * state_dim
    state_update = state_dim * observation_dim
    joseph = (
        state_dim * observation_dim * state_dim
        + 2 * state_dim**3
        + state_dim * observation_dim**2
        + state_dim**2 * observation_dim
    )
    kalman_macs = moment_postprocess + prediction + innovation + solve + state_update + joseph
    # The integer division in n^3//3 drops a fraction; retain a conservative
    # two-MAC ceiling for the small 5x5 solve.
    kalman_macs += 2
    hmm, _, _ = _load_hmm()
    classes = len(REGIME_CLASSES)
    features = int(hmm.standardization_mean.size)
    history = 8
    hmm_macs = classes * features * features + features + history * classes * classes
    kalman_state_values = 10 + 100 + 100 + 50 + 100 + 25 + 50
    kalman_state_bytes = kalman_state_values * 4
    hmm_state_bytes = (hmm.parameter_count + history * classes + classes) * 4
    route_macs = kalman_macs + hmm_macs
    route_state_bytes = kalman_state_bytes + hmm_state_bytes
    return {
        "standard_binning": MethodCost(0, 0, 0, "fixed central-class rule"),
        "static_joint_map": MethodCost(0, 6 * 4, 256, "two means plus symmetric 2x2 covariance"),
        "window_map": MethodCost(moment_postprocess, 4 * 2 * 4, 512, "four complex moments plus postprocess"),
        "ewma_adaptive_map": MethodCost(moment_postprocess + 4 * 2, (4 * 2 + 12) * 4, 512, "window postprocess plus four complex blends"),
        "kalman_adaptive_map": MethodCost(kalman_macs, kalman_state_bytes, 2048, "dense 10-state Joseph-form NumPy path including 5x5 solve"),
        "proposed_route_a": MethodCost(route_macs, route_state_bytes, 3072, "worst same-cycle Kalman parameter update plus rolling-HMM regime update"),
    }


def _load_static_and_hyperparameters() -> tuple[StaticMAPParameters, dict[str, float]]:
    report = json.loads(_repo_path(PARENT_MIXED_REPORT).read_text(encoding="utf-8"))
    lane = report["decoder_lane"]
    static = StaticMAPParameters(**lane["static_training_parameters"])
    frozen = lane["frozen_hyperparameters"]
    return static, {
        "ewma_alpha": float(frozen["ewma_alpha"]),
        "kalman_process_scale": float(frozen["kalman_process_scale"]),
        "kalman_measurement_scale": float(frozen["kalman_measurement_scale"]),
    }


def _load_hmm() -> tuple[GaussianRegimeHMM, float, dict[str, Any]]:
    path = _repo_path(HMM_CHECKPOINT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "t6.6.1-exported-t4.1.1-gaussian-hmm-v1":
        raise ValueError("exported HMM schema mismatch")
    source = _repo_path(payload["source_checkpoint"])
    if _sha256(source) != payload["source_checkpoint_sha256"]:
        raise ValueError("exported HMM source checkpoint hash mismatch")
    if payload.get("selected_family_from_validation") != "gaussian_hmm":
        raise ValueError("exported HMM was not the validation-selected family")
    model_payload = payload["model"]
    array_fields = (
        "standardization_mean",
        "standardization_scale",
        "emission_means",
        "emission_covariances",
        "emission_precisions",
        "emission_log_determinants",
        "transition_matrix",
        "initial_probabilities",
        "class_prior_probabilities",
    )
    model = GaussianRegimeHMM(
        **{field: np.asarray(model_payload[field], dtype=np.float64) for field in array_fields},
        covariance_regularization=float(model_payload["covariance_regularization"]),
        transition_smoothing=float(model_payload["transition_smoothing"]),
    )
    return model, float(payload["temperature"]), payload


def _packet_payload(
    *,
    trace_id: str,
    cycle: int,
    syndrome_code: int,
    phase: int,
    selected_residual: float,
) -> dict[str, object]:
    threshold = 0.35 * LATTICE_CONST
    event = "e" if abs(selected_residual) >= threshold else "g"
    normalized = min(1.0, abs(selected_residual) / (0.5 * LATTICE_CONST))
    return {
        "schema_id": OBSERVED_SCHEMA_ID,
        "trace_id": trace_id,
        "cycle_index": cycle,
        "syndrome_code": int(syndrome_code),
        "syndrome_x": event if phase == 1 else "g",
        "syndrome_z": event if phase == 0 else "g",
        "quadrature_phase_bit": phase,
        "ood_score_code": int(round(255.0 * normalized)),
        "parameter_age_code": cycle % MatchedBudget().parameter_update_period_cycles,
        "reset_ack": False,
        "observation_valid": True,
        "deadline_ok": True,
    }


def _quantize_residuals(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    config = ParametricMAPLUTConfig()
    codes = np.empty_like(residuals, dtype=np.int64)
    decoded = np.empty_like(residuals, dtype=np.float64)
    for phase in (0, 1):
        phase_codes = np.fromiter(
            (software_encode_syndrome_for_replay(value, config) for value in residuals[:, phase]),
            dtype=np.int64,
            count=len(residuals),
        )
        codes[:, phase] = phase_codes
        decoded[:, phase] = np.fromiter(
            (software_decode_syndrome_code(int(code), config) for code in phase_codes),
            dtype=np.float64,
            count=len(phase_codes),
        )
    return codes, decoded


def _pack_window(
    trace_id: str,
    window_id: int,
    start_cycle: int,
    codes: np.ndarray,
    decoded: np.ndarray,
) -> ObservedTraceWindow:
    packets: list[ObservedSyndromePacket] = []
    for pair_id in range(len(codes)):
        for phase in (0, 1):
            payload = _packet_payload(
                trace_id=trace_id,
                cycle=start_cycle + 2 * pair_id + phase,
                syndrome_code=int(codes[pair_id, phase]),
                phase=phase,
                selected_residual=float(decoded[pair_id, phase]),
            )
            packet = validate_observed_mapping_for_deployable("standard_binning", payload)
            validate_wire_roundtrip(packet)
            packets.append(packet)
    packet_tuple = tuple(packets)
    return ObservedTraceWindow(
        window_id=window_id,
        packets=packet_tuple,
        packet_sha256=_canonical_sha256([asdict(item) for item in packet_tuple]),
    )


def materialize_qualification_trace(
    config: RunnerConfig | None = None,
) -> tuple[ObservedTrace, IsolatedTruthTrace]:
    settings = RunnerConfig() if config is None else config
    if not isinstance(settings, RunnerConfig):
        raise TypeError("config must be RunnerConfig")
    scenario = MixedDecoderScenario(settings.scenario_id)
    states = scenario.states(settings.windows)
    rng = np.random.default_rng(settings.seed)
    calibration_values: list[np.ndarray] = []
    for _ in range(settings.calibration_windows):
        residual, _, _ = _residuals_and_truth(
            states[0], settings.pairs_per_parameter_window, rng
        )
        _, decoded = _quantize_residuals(residual)
        calibration_values.append(decoded)
    calibration = np.concatenate(calibration_values, axis=0)
    calibration_sha = _array_sha256(calibration, "<f8")
    trace_id = f"t661-{settings.scenario_id}-{settings.seed}"
    observed_windows: list[ObservedTraceWindow] = []
    truth_windows: list[TruthTraceWindow] = []
    for window_id, state in enumerate(states):
        residual, truth, _ = _residuals_and_truth(
            state, settings.pairs_per_parameter_window, rng
        )
        codes, decoded = _quantize_residuals(residual)
        start_cycle = 1 + window_id * settings.parameter_window_stride_cycles
        observed_windows.append(
            _pack_window(trace_id, window_id, start_cycle, codes, decoded)
        )
        truth_windows.append(
            TruthTraceWindow(
                window_id=window_id,
                truth=truth,
                state=state,
                truth_sha256=_array_sha256(truth, "<i8"),
            )
        )
    window_hashes = [item.packet_sha256 for item in observed_windows]
    trace_hash = _canonical_sha256(
        {
            "trace_id": trace_id,
            "calibration_sha256": calibration_sha,
            "window_hashes": window_hashes,
        }
    )
    return (
        ObservedTrace(
            trace_id=trace_id,
            calibration_residuals=calibration,
            calibration_sha256=calibration_sha,
            windows=tuple(observed_windows),
            trace_sha256=trace_hash,
        ),
        IsolatedTruthTrace(trace_id=trace_id, windows=tuple(truth_windows)),
    )


def paired_residuals_from_packets(
    packets: Sequence[ObservedSyndromePacket],
    *,
    method_id: str,
) -> np.ndarray:
    """Rebuild joint syndromes from the canonical wire input and nothing else."""

    sequence = tuple(packets)
    if len(sequence) % 2:
        raise ContractViolation(
            "incomplete_phase_pair",
            field="packets",
            expected="even packet count",
            actual=len(sequence),
            method_id=method_id,
        )
    config = ParametricMAPLUTConfig()
    output = np.empty((len(sequence) // 2, 2), dtype=np.float64)
    for pair_id in range(len(output)):
        q_packet, p_packet = sequence[2 * pair_id : 2 * pair_id + 2]
        for packet in (q_packet, p_packet):
            validate_observed_mapping_for_deployable(method_id, asdict(packet))
            validate_wire_roundtrip(packet)
        if q_packet.trace_id != p_packet.trace_id:
            raise ContractViolation(
                "phase_pair_trace_mismatch",
                field="trace_id",
                expected=q_packet.trace_id,
                actual=p_packet.trace_id,
                method_id=method_id,
            )
        if (q_packet.quadrature_phase_bit, p_packet.quadrature_phase_bit) != (0, 1):
            raise ContractViolation(
                "phase_pair_order_mismatch",
                field="quadrature_phase_bit",
                expected=(0, 1),
                actual=(q_packet.quadrature_phase_bit, p_packet.quadrature_phase_bit),
                method_id=method_id,
            )
        if p_packet.cycle_index != q_packet.cycle_index + 1:
            raise ContractViolation(
                "phase_pair_not_adjacent",
                field="cycle_index",
                expected=q_packet.cycle_index + 1,
                actual=p_packet.cycle_index,
                method_id=method_id,
            )
        output[pair_id, 0] = software_decode_syndrome_code(q_packet.syndrome_code, config)
        output[pair_id, 1] = software_decode_syndrome_code(p_packet.syndrome_code, config)
    return output


def _accounting_record(
    method_id: str,
    trace_id: str,
    cycle: int,
    *,
    update_due: bool,
    update_macs: int,
    state_bytes: int,
    workspace_bytes: int,
    wallclock_us: float,
) -> ExecutionAccountingRecord:
    budget = MatchedBudget()
    record = ExecutionAccountingRecord(
        method_id=method_id,
        trace_id=trace_id,
        cycle_index=cycle,
        action_valid_cycle=cycle + budget.fast_action_latency_cycles,
        source_to_action_cycles=budget.fast_action_latency_cycles,
        logical_deadline_miss=False,
        update_due=update_due,
        update_macs=update_macs,
        private_model_state_bytes=state_bytes,
        transient_workspace_bytes=workspace_bytes,
        host_update_wallclock_us=float(wallclock_us),
        host_update_deadline_miss=wallclock_us > budget.max_host_update_wallclock_us,
        board_measured_deadline_miss=None,
    )
    assert_accounting_conforms(record, budget)
    return record


@dataclass(frozen=True)
class RouteWindowResult:
    decisions: np.ndarray
    posterior: np.ndarray
    reason_counts: dict[str, int]
    last_regime_posterior: tuple[float, float, float, float]
    hmm_updates: int
    parameter_update_applied: bool
    max_hmm_step_us: float
    parameter_update_us: float


class RouteAIntegrationAdapter:
    """Executable pre-policy-lock integration adapter.

    It exercises the real HMM and Kalman candidate on observed packets.  The
    thresholds are intentionally marked integration-only; T6.6.2/T6.6.3 own
    full fault-state semantics and pilot-only threshold freezing.
    """

    def __init__(
        self,
        calibration: np.ndarray,
        static: StaticMAPParameters,
        hyperparameters: Mapping[str, float],
        config: RunnerConfig,
    ) -> None:
        self.static = static
        self.config = config
        self.moment = PeriodicMomentConfig(minimum_samples=64)
        self.kalman = ConstantVelocityPeriodicKalman(
            calibration,
            moment_config=self.moment,
            kalman_config=scaled_periodic_kalman_config(
                process_scale=hyperparameters["kalman_process_scale"],
                measurement_scale=hyperparameters["kalman_measurement_scale"],
            ),
        )
        hmm, temperature, _ = _load_hmm()
        # Warm NumPy's optimized einsum-path cache before wall-clock accounting;
        # this does not advance the causal filter or consume an observation.
        hmm.emission_log_likelihood(np.zeros((1, len(hmm.standardization_mean))))
        self.hmm = RollingGaussianHMMAdapter(
            hmm, history_windows=8, temperature=temperature
        )
        self._last_q = 0.0
        self._last_p = 0.0
        self._regime_posterior: np.ndarray | None = None

    def run_window(
        self,
        window: ObservedTraceWindow,
    ) -> RouteWindowResult:
        residuals = paired_residuals_from_packets(
            window.packets, method_id="proposed_route_a"
        )
        static_result = map_decode_2d(
            residuals,
            self.static.covariance_array(),
            mean=self.static.mean_array(),
        )
        kalman_prediction = self.kalman.prediction()
        kalman_result = map_decode_2d(
            residuals,
            kalman_prediction.covariance_array(),
            mean=kalman_prediction.mean_array(),
        )
        static_decisions = np.asarray(static_result.logical_class, dtype=np.int64)
        kalman_decisions = np.asarray(kalman_result.logical_class, dtype=np.int64)
        static_posterior = np.asarray(static_result.posterior, dtype=np.float64).reshape((-1, 4))
        kalman_posterior = np.asarray(kalman_result.posterior, dtype=np.float64).reshape((-1, 4))
        decisions = np.empty_like(static_decisions)
        posterior = np.empty_like(static_posterior)
        raw_rows: list[list[float]] = []
        reasons: Counter[str] = Counter()
        max_hmm_step_us = 0.0
        hmm_updates = 0
        severe_health_fault = False
        for packet_index, packet in enumerate(window.packets):
            value = software_decode_syndrome_code(
                packet.syndrome_code, ParametricMAPLUTConfig()
            )
            if packet.quadrature_phase_bit == 0:
                self._last_q = value
            else:
                self._last_p = value
            raw_rows.append(
                [
                    self._last_q,
                    self._last_p,
                    float(packet.syndrome_x == "e"),
                    float(packet.syndrome_z == "e"),
                    float(packet.syndrome_x == "leakage" or packet.syndrome_z == "leakage"),
                    float(packet.quadrature_phase_bit),
                    float(packet.observation_valid),
                    float(packet.deadline_ok),
                ]
            )
            severe_health_fault = severe_health_fault or (
                not packet.observation_valid
                or not packet.deadline_ok
                or packet.syndrome_x == "leakage"
                or packet.syndrome_z == "leakage"
            )
            if len(raw_rows) == self.config.regime_window_cycles:
                observation = RegimeObservationWindow(
                    window_index=window.window_id * 64 + hmm_updates,
                    start_cycle=packet.cycle_index - len(raw_rows) + 1,
                    values=np.asarray(raw_rows, dtype=np.float64),
                )
                summary = summarize_regime_window(observation)
                start = perf_counter_ns()
                self._regime_posterior = self.hmm.step(summary)
                elapsed = (perf_counter_ns() - start) / 1000.0
                max_hmm_step_us = max(max_hmm_step_us, elapsed)
                hmm_updates += 1
                raw_rows.clear()
            if packet.quadrature_phase_bit != 1:
                continue
            pair_id = packet_index // 2
            if self._regime_posterior is None:
                choice = "hmm_warmup_trusted_static"
                use_kalman = False
            else:
                regime = {
                    name: float(self._regime_posterior[index])
                    for index, name in enumerate(REGIME_CLASSES)
                }
                if severe_health_fault:
                    choice = "health_fail_closed_trusted_static"
                    use_kalman = False
                elif regime["burst"] + regime["calibration_shift"] >= 0.60:
                    choice = "tail_regime_trusted_static"
                    use_kalman = False
                elif regime["normal"] >= 0.55 and max(regime.values()) >= 0.55:
                    choice = "normal_kalman_candidate"
                    use_kalman = True
                else:
                    choice = "posterior_uncertain_trusted_static"
                    use_kalman = False
            reasons[choice] += 1
            decisions[pair_id] = kalman_decisions[pair_id] if use_kalman else static_decisions[pair_id]
            posterior[pair_id] = kalman_posterior[pair_id] if use_kalman else static_posterior[pair_id]
        if raw_rows:
            raise RuntimeError("regime windows did not close exactly on the packet cadence")
        final = self._regime_posterior
        allow_update = bool(
            final is not None
            and float(final[REGIME_CLASSES.index("normal")]) >= 0.55
            and not severe_health_fault
        )
        start = perf_counter_ns()
        if allow_update:
            self.kalman.update(residuals, window_id=window.window_id)
        parameter_update_us = (perf_counter_ns() - start) / 1000.0
        normalized = (
            np.asarray((0.25, 0.25, 0.25, 0.25), dtype=np.float64)
            if final is None
            else np.asarray(final, dtype=np.float64)
        )
        return RouteWindowResult(
            decisions=decisions,
            posterior=posterior,
            reason_counts=dict(sorted(reasons.items())),
            last_regime_posterior=tuple(float(value) for value in normalized),
            hmm_updates=hmm_updates,
            parameter_update_applied=allow_update,
            max_hmm_step_us=max_hmm_step_us,
            parameter_update_us=parameter_update_us,
        )


def _prediction_result(residuals: np.ndarray, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    result = map_decode_2d(
        residuals,
        prediction.covariance_array(),
        mean=prediction.mean_array(),
    )
    return (
        np.asarray(result.logical_class, dtype=np.int64),
        np.asarray(result.posterior, dtype=np.float64).reshape((-1, 4)),
    )


def _fixed_standard_posterior(samples: int) -> np.ndarray:
    posterior = np.zeros((samples, 4), dtype=np.float64)
    posterior[:, 0] = 1.0
    return posterior


def run_deployable_methods(
    observed: ObservedTrace,
    config: RunnerConfig | None = None,
) -> dict[str, MethodRun]:
    settings = RunnerConfig() if config is None else config
    static, hyper = _load_static_and_hyperparameters()
    moment = PeriodicMomentConfig(minimum_samples=64)
    latest = LatestWindowPeriodicPredictor(observed.calibration_residuals, moment)
    ewma = PeriodicMomentEWMA(
        observed.calibration_residuals,
        alpha=hyper["ewma_alpha"],
        config=moment,
    )
    kalman = ConstantVelocityPeriodicKalman(
        observed.calibration_residuals,
        moment_config=moment,
        kalman_config=scaled_periodic_kalman_config(
            process_scale=hyper["kalman_process_scale"],
            measurement_scale=hyper["kalman_measurement_scale"],
        ),
    )
    route = RouteAIntegrationAdapter(
        observed.calibration_residuals, static, hyper, settings
    )
    decisions: dict[str, list[np.ndarray]] = {name: [] for name in COMMON_TRACE_METHODS}
    posteriors: dict[str, list[np.ndarray]] = {name: [] for name in COMMON_TRACE_METHODS}
    accounting: dict[str, list[ExecutionAccountingRecord]] = {
        name: [] for name in COMMON_TRACE_METHODS
    }
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in COMMON_TRACE_METHODS}
    cost = derive_method_costs()
    for window in observed.windows:
        residuals_by_method = {
            method_id: paired_residuals_from_packets(window.packets, method_id=method_id)
            for method_id in COMMON_TRACE_METHODS
        }
        hashes = {
            _array_sha256(value, "<f8") for value in residuals_by_method.values()
        }
        if len(hashes) != 1:
            raise RuntimeError("deployable adapters reconstructed different syndrome inputs")
        residuals = residuals_by_method["standard_binning"]
        samples = len(residuals)
        standard_decision = np.zeros(samples, dtype=np.int64)
        standard_posterior = _fixed_standard_posterior(samples)
        static_result = map_decode_2d(
            residuals,
            static.covariance_array(),
            mean=static.mean_array(),
        )
        static_decision = np.asarray(static_result.logical_class, dtype=np.int64)
        static_posterior = np.asarray(static_result.posterior, dtype=np.float64).reshape((-1, 4))
        latest_decision, latest_posterior = _prediction_result(residuals, latest.prediction())
        ewma_decision, ewma_posterior = _prediction_result(residuals, ewma.prediction())
        kalman_decision, kalman_posterior = _prediction_result(residuals, kalman.prediction())
        route_result = route.run_window(window)
        outputs = {
            "standard_binning": (standard_decision, standard_posterior),
            "static_joint_map": (static_decision, static_posterior),
            "window_map": (latest_decision, latest_posterior),
            "ewma_adaptive_map": (ewma_decision, ewma_posterior),
            "kalman_adaptive_map": (kalman_decision, kalman_posterior),
            "proposed_route_a": (route_result.decisions, route_result.posterior),
        }
        update_elapsed: dict[str, float] = {
            "standard_binning": 0.0,
            "static_joint_map": 0.0,
        }
        for method_id, predictor in (
            ("window_map", latest),
            ("ewma_adaptive_map", ewma),
            ("kalman_adaptive_map", kalman),
        ):
            start = perf_counter_ns()
            predictor.update(residuals, window_id=window.window_id)
            update_elapsed[method_id] = (perf_counter_ns() - start) / 1000.0
        update_elapsed["proposed_route_a"] = (
            route_result.max_hmm_step_us
            + (
                route_result.parameter_update_us
                if route_result.parameter_update_applied
                else 0.0
            )
        )
        last_cycle = window.packets[-1].cycle_index
        for method_id, (method_decisions, method_posterior) in outputs.items():
            decisions[method_id].append(method_decisions)
            posteriors[method_id].append(method_posterior)
            method_cost = cost[method_id]
            macs = method_cost.update_macs
            state_bytes = method_cost.private_model_state_bytes
            workspace = method_cost.transient_workspace_bytes
            update_due = method_id not in ("standard_binning", "static_joint_map")
            record_macs = macs if update_due else 0
            total_window_macs = record_macs
            if method_id == "proposed_route_a":
                kalman_macs = cost["kalman_adaptive_map"].update_macs
                hmm_macs = cost["proposed_route_a"].update_macs - kalman_macs
                record_macs = hmm_macs + (
                    kalman_macs if route_result.parameter_update_applied else 0
                )
                total_window_macs = route_result.hmm_updates * hmm_macs + (
                    kalman_macs if route_result.parameter_update_applied else 0
                )
            accounting[method_id].append(
                _accounting_record(
                    method_id,
                    observed.trace_id,
                    last_cycle,
                    update_due=update_due,
                    update_macs=record_macs,
                    state_bytes=state_bytes,
                    workspace_bytes=workspace,
                    wallclock_us=update_elapsed[method_id],
                )
            )
            row = {
                "method_id": method_id,
                "window_id": window.window_id,
                "input_sha256": next(iter(hashes)),
                "decision_sha256": _array_sha256(method_decisions, "<i8"),
                "decision_nonzero_count": int(np.count_nonzero(method_decisions)),
                "update_wallclock_us": update_elapsed[method_id],
                "update_macs": record_macs,
                "total_window_update_macs": total_window_macs,
                "private_model_state_bytes": state_bytes,
                "transient_workspace_bytes": workspace,
                "cost_derivation": method_cost.derivation,
            }
            if method_id == "proposed_route_a":
                row.update(
                    {
                        "route_reason_counts": route_result.reason_counts,
                        "last_regime_posterior": list(route_result.last_regime_posterior),
                        "hmm_updates": route_result.hmm_updates,
                        "parameter_update_applied": route_result.parameter_update_applied,
                    }
                )
            rows[method_id].append(row)
    manifests = {row.method_id: row for row in canonical_method_manifests()}
    return {
        method_id: MethodRun(
            method_id=method_id,
            trace_sha256=observed.trace_sha256,
            decisions=tuple(decisions[method_id]),
            posteriors=tuple(posteriors[method_id]),
            accounting=tuple(accounting[method_id]),
            window_rows=tuple(rows[method_id]),
            private_model_state_bytes=cost[method_id].private_model_state_bytes,
            max_update_macs=cost[method_id].update_macs,
            max_transient_workspace_bytes=cost[method_id].transient_workspace_bytes,
            actual_execution=(
                "real_frozen_rule"
                if method_id == "standard_binning"
                else "real_joint_map_decode"
                if method_id == "static_joint_map"
                else "real_periodic_predictor_then_causal_update"
                if method_id in ("window_map", "ewma_adaptive_map", "kalman_adaptive_map")
                else "real_hmm_plus_kalman_integration_thresholds_not_locked"
            ),
            current_rtl_status=manifests[method_id].current_rtl_compatibility,
            ranking_status=(
                "integration_qualification_only_not_formal_result"
                if method_id == "proposed_route_a"
                else "eligible_for_common_trace_qualification"
            ),
        )
        for method_id in COMMON_TRACE_METHODS
    }


def _oracle_run(
    observed: ObservedTrace,
    truth: IsolatedTruthTrace,
) -> dict[str, Any]:
    if observed.trace_id != truth.trace_id or len(observed.windows) != len(truth.windows):
        raise ValueError("oracle trace alignment failed")
    decisions: list[np.ndarray] = []
    posteriors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for observed_window, truth_window in zip(observed.windows, truth.windows, strict=True):
        residuals = paired_residuals_from_packets(
            observed_window.packets, method_id="static_joint_map"
        )
        result = oracle_map_2d(residuals, truth_window.state)
        decision = np.asarray(result.logical_class, dtype=np.int64)
        posterior = np.asarray(result.posterior, dtype=np.float64).reshape((-1, 4))
        decisions.append(decision)
        posteriors.append(posterior)
        rows.append(
            {
                "window_id": observed_window.window_id,
                "decision_sha256": _array_sha256(decision, "<i8"),
                "privileged_state_step": truth_window.state.step,
                "privileged_regime": truth_window.state.regime,
            }
        )
    digest = hashlib.sha256()
    for decision in decisions:
        digest.update(np.asarray(decision, dtype="<i8").tobytes())
    return {
        "manifest": oracle_method_manifest().to_dict(),
        "trace_sha256": observed.trace_sha256,
        "decision_sha256": digest.hexdigest(),
        "decisions": tuple(decisions),
        "posteriors": tuple(posteriors),
        "window_rows": rows,
        "physical_table": "oracle_upper_bound_only",
        "included_in_deployable_ranking": False,
        "accounting": None,
    }


def _legacy_cnn_witness(config: RunnerConfig) -> dict[str, Any]:
    parent = json.loads(_repo_path(CNN_PARENT_REPORT).read_text(encoding="utf-8"))
    anchors = {
        row["binding_id"]: row for row in parent["cnn_asset_bindings"]
    }
    expected_paths = {
        "model": CNN_MODEL,
        "manifest": CNN_MANIFEST,
        "test_split": CNN_TEST_SPLIT,
    }
    for binding_id, expected_path in expected_paths.items():
        anchor = anchors.get(binding_id)
        if not isinstance(anchor, Mapping):
            raise ValueError(f"T5.4.3 has no {binding_id} CNN anchor")
        if anchor["path"] != expected_path.as_posix() or anchor["sha256"] != _sha256(expected_path):
            raise ValueError(f"legacy CNN {binding_id} drifted from the T5.4.3 anchor")
    manifest = json.loads(_repo_path(CNN_MANIFEST).read_text(encoding="utf-8"))
    with np.load(_repo_path(CNN_TEST_SPLIT), allow_pickle=True) as data:
        histograms = np.asarray(data["histograms"][: config.cnn_witness_samples], dtype=np.float32)
        available_keys = tuple(sorted(data.files))
    if histograms.shape[1:] != tuple(manifest["histogram_shape"]):
        raise ValueError("legacy CNN witness shape disagrees with its manifest")
    start = perf_counter_ns()
    prediction = np.asarray(
        predict_from_artifact(_repo_path(CNN_MODEL), histograms), dtype=np.float64
    )
    wallclock_us = (perf_counter_ns() - start) / 1000.0
    with np.load(_repo_path(CNN_MODEL), allow_pickle=True) as model:
        conv_w = np.asarray(model["conv_w"])
        fc1_w = np.asarray(model["fc1_w"])
        fc2_w = np.asarray(model["fc2_w"])
        inference_arrays = (
            "conv_w",
            "conv_b",
            "fc1_w",
            "fc1_b",
            "fc2_w",
            "fc2_b",
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
        )
        state_bytes = int(sum(np.asarray(model[name]).nbytes for name in inference_arrays))
    output_height, output_width = int(manifest["histogram_shape"][1]), int(manifest["histogram_shape"][2])
    conv_macs = int(
        output_height
        * output_width
        * conv_w.shape[0]
        * conv_w.shape[1]
        * conv_w.shape[2]
        * conv_w.shape[3]
    )
    macs = conv_macs + int(fc1_w.shape[0] * fc1_w.shape[1]) + int(fc2_w.shape[0] * fc2_w.shape[1])
    workspace = max(
        int(conv_w.shape[0] * output_height * output_width * 4),
        int(fc1_w.shape[0] * 4),
    )
    budget = MatchedBudget()
    violations = {
        "input_schema": (
            "legacy_21_channel_native_tensor_contains_teacher_channels_and_has_no_"
            "truth_free_transform_from_the_T6.5.2_scalar_packet_schema"
        ),
        "update_macs": {"actual": macs, "limit": budget.max_algorithm_macs_per_parameter_update},
        "private_model_state_bytes": {"actual": state_bytes, "limit": budget.max_private_model_state_bytes},
        "transient_workspace_bytes": {"actual": workspace, "limit": budget.max_transient_workspace_bytes},
    }
    return {
        "method_id": "legacy_cnn_residual",
        "actual_checkpoint_loaded": True,
        "checkpoint_path": CNN_MODEL.as_posix(),
        "checkpoint_sha256": _sha256(CNN_MODEL),
        "manifest_path": CNN_MANIFEST.as_posix(),
        "manifest_sha256": _sha256(CNN_MANIFEST),
        "test_split_path": CNN_TEST_SPLIT.as_posix(),
        "test_split_sha256": _sha256(CNN_TEST_SPLIT),
        "t5_4_3_anchor_report": CNN_PARENT_REPORT.as_posix(),
        "t5_4_3_anchor_report_sha256": _sha256(CNN_PARENT_REPORT),
        "all_three_assets_match_t5_4_3_anchor": True,
        "witness_samples": len(histograms),
        "witness_input_sha256": _array_sha256(histograms, "<f4"),
        "witness_prediction_sha256": _array_sha256(prediction, "<f8"),
        "witness_wallclock_us_total": wallclock_us,
        "available_native_split_keys": list(available_keys),
        "adapter_consumed_keys": ["histograms"],
        "labels_or_target_consumed_online": False,
        "macs_per_inference": macs,
        "private_model_state_bytes": state_bytes,
        "transient_workspace_bytes_lower_bound": workspace,
        "common_trace_sha256": None,
        "input_schema_conforms": False,
        "matched_budget_conforms": False,
        "violations": violations,
        "ranking_status": "ablation_only_failed_input_schema_and_matched_budget",
        "metric_lane": "legacy_native_residual_b_witness_not_LER_comparison",
    }


def _standard_lut_equivalence() -> dict[str, Any]:
    config = ParametricMAPLUTConfig()
    code = 1 << config.llr_fractional_bits
    source_hash = hashlib.sha256(b"standard-binning-fixed-central-class-v1").hexdigest()
    image = ParametricMAPLUTImage.create(
        config=config,
        active_bank_version=0,
        source_params_sha256=source_hash,
        model_mean=(0.0, 0.0),
        model_sigma=(1.0, 1.0),
        table_codes=(
            tuple(code for _ in range(config.table_entries)),
            tuple(code for _ in range(config.table_entries)),
        ),
        llr_saturation_count=0,
    )
    runtime = ParametricMAPLUTRuntime(image)
    mismatches = 0
    for phase in (0, 1):
        for syndrome_code in range(config.adc_levels):
            decision = runtime.decode_code(
                ParametricMAPLUTInput(
                    cycle_index=syndrome_code + phase * config.adc_levels,
                    syndrome_code=syndrome_code,
                    quadrature_phase_bit=phase,
                    active_bank_version=0,
                )
            )
            mismatches += int(decision.logical_flip or decision.llr_code != code)
    return {
        "image_crc32": image.image_crc32,
        "image_sha256": image.image_sha256,
        "phase_code_cases": 2 * config.adc_levels,
        "mismatches": mismatches,
        "logical_rule": "centered_modular_syndrome_always_selects_central_even_class",
        "current_rtl_scope": "phase_action_equivalence_only_not_full_2d_joint_map",
    }


def _periodic_feature_grid_equivalence() -> dict[str, Any]:
    """Exhaustively prove product-free characteristic features on the ADC grid."""

    levels = ParametricMAPLUTConfig().adc_levels
    codes = np.arange(levels, dtype=np.int64)
    phase = np.exp(1j * (-math.pi + 2.0 * math.pi * (codes + 0.5) / levels))
    index_lut = np.exp(1j * 2.0 * math.pi * codes / levels)
    q = codes[:, None]
    p = codes[None, :]
    product_sum = phase[:, None] * phase[None, :]
    product_difference = phase[:, None] * np.conjugate(phase[None, :])
    lookup_sum = index_lut[(q + p + 1) % levels]
    lookup_difference = index_lut[(q - p) % levels]
    max_error = max(
        float(np.max(np.abs(product_sum - lookup_sum))),
        float(np.max(np.abs(product_difference - lookup_difference))),
    )
    return {
        "adc_levels": levels,
        "exhaustive_qp_pairs": levels * levels,
        "checked_complex_product_identities": 2 * levels * levels,
        "max_absolute_complex_error": max_error,
        "tolerance": 5.0e-12,
        "mapping": {
            "q_phase": "phase_lut[q_code]",
            "p_phase": "phase_lut[p_code]",
            "q_plus_p": "index_lut[(q_code+p_code+1) mod 1024]",
            "q_minus_p": "index_lut[(q_code-p_code) mod 1024]",
        },
        "deployment_operation_class": "four_LUT_reads_two_integer_indices_integer_accumulation_no_real_MAC",
    }


def _future_mutated_trace(observed: ObservedTrace, prefix_windows: int) -> ObservedTrace:
    windows = list(observed.windows)
    for index in range(prefix_windows, len(windows)):
        source = windows[index]
        mutated_packets = []
        for packet in source.packets:
            mutated_packets.append(
                replace(
                    packet,
                    syndrome_code=1023 - packet.syndrome_code,
                    ood_score_code=255 - packet.ood_score_code,
                )
            )
        packet_tuple = tuple(mutated_packets)
        windows[index] = ObservedTraceWindow(
            window_id=source.window_id,
            packets=packet_tuple,
            packet_sha256=_canonical_sha256([asdict(item) for item in packet_tuple]),
        )
    trace_hash = _canonical_sha256(
        {
            "trace_id": observed.trace_id,
            "calibration_sha256": observed.calibration_sha256,
            "window_hashes": [item.packet_sha256 for item in windows],
        }
    )
    return ObservedTrace(
        trace_id=observed.trace_id,
        calibration_residuals=observed.calibration_residuals,
        calibration_sha256=observed.calibration_sha256,
        windows=tuple(windows),
        trace_sha256=trace_hash,
    )


def _prefix_causality(
    observed: ObservedTrace,
    original: Mapping[str, MethodRun],
    config: RunnerConfig,
) -> dict[str, Any]:
    mutated_trace = _future_mutated_trace(observed, config.prefix_windows)
    mutated = run_deployable_methods(mutated_trace, config)
    rows = []
    for method_id in COMMON_TRACE_METHODS:
        original_prefix = np.concatenate(original[method_id].decisions[: config.prefix_windows])
        mutated_prefix = np.concatenate(mutated[method_id].decisions[: config.prefix_windows])
        future_original = np.concatenate(original[method_id].decisions[config.prefix_windows :])
        future_mutated = np.concatenate(mutated[method_id].decisions[config.prefix_windows :])
        rows.append(
            {
                "method_id": method_id,
                "prefix_equal": bool(np.array_equal(original_prefix, mutated_prefix)),
                "future_input_changed": (
                    observed.trace_sha256 != mutated_trace.trace_sha256
                ),
                "future_decision_changed": bool(
                    not np.array_equal(future_original, future_mutated)
                ),
            }
        )
    return {
        "prefix_windows": config.prefix_windows,
        "mutated_trace_sha256": mutated_trace.trace_sha256,
        "rows": rows,
    }


def _evaluate_runs(
    runs: Mapping[str, MethodRun],
    oracle: Mapping[str, Any],
    truth: IsolatedTruthTrace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    window_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    all_methods: dict[str, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = {
        method_id: (run.decisions, run.posteriors) for method_id, run in runs.items()
    }
    all_methods["hidden_state_oracle"] = (
        oracle["decisions"],
        oracle["posteriors"],
    )
    for method_id, (decisions, posteriors) in all_methods.items():
        failures = 0
        total = 0
        nll_weighted = 0.0
        brier_weighted = 0.0
        for truth_window, decision, posterior in zip(
            truth.windows, decisions, posteriors, strict=True
        ):
            actual = np.asarray(decision, dtype=np.int64)
            target = truth_window.truth
            nll, brier = _mean_nll_brier(posterior, target)
            window_failures = int(np.count_nonzero(actual != target))
            failures += window_failures
            total += len(target)
            nll_weighted += nll * len(target)
            brier_weighted += brier * len(target)
            window_rows.append(
                {
                    "method_id": method_id,
                    "window_id": truth_window.window_id,
                    "samples": len(target),
                    "failures": window_failures,
                    "logical_error_rate": window_failures / len(target),
                    "nll": nll,
                    "brier": brier,
                }
            )
        method_rows.append(
            {
                "method_id": method_id,
                "samples": total,
                "failures": failures,
                "logical_error_rate": failures / total,
                "nll": nll_weighted / total,
                "brier": brier_weighted / total,
                "table": (
                    "oracle_upper_bound_only"
                    if method_id == "hidden_state_oracle"
                    else "deployable_common_trace_qualification"
                ),
            }
        )
    return method_rows, window_rows


def _mutation_audit(observed: ObservedTrace, cnn: Mapping[str, Any]) -> list[dict[str, Any]]:
    first = observed.windows[0]
    cases: list[tuple[str, Any]] = []

    def record(case_id: str, operation: Any) -> None:
        try:
            operation()
        except Exception as exc:  # noqa: BLE001 - audit records exact rejection.
            cases.append(
                {
                    "case_id": case_id,
                    "rejected": True,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )
        else:
            cases.append(
                {"case_id": case_id, "rejected": False, "exception": None, "message": None}
            )

    record(
        "phase_order_swapped",
        lambda: paired_residuals_from_packets(
            (first.packets[1], first.packets[0], *first.packets[2:]),
            method_id="static_joint_map",
        ),
    )
    record(
        "phase_cycle_not_adjacent",
        lambda: paired_residuals_from_packets(
            (first.packets[0], replace(first.packets[1], cycle_index=999999), *first.packets[2:]),
            method_id="static_joint_map",
        ),
    )
    record(
        "incomplete_phase_pair",
        lambda: paired_residuals_from_packets(first.packets[:-1], method_id="static_joint_map"),
    )
    record(
        "hidden_truth_side_channel",
        lambda: validate_observed_mapping_for_deployable(
            "kalman_adaptive_map", {**asdict(first.packets[0]), "logical_truth": 0}
        ),
    )
    record(
        "cnn_missing_checkpoint",
        lambda: predict_from_artifact(ROOT / "missing-cnn-checkpoint.npz", np.zeros((1, 21, 32, 32))),
    )

    def cnn_budget() -> None:
        record_value = ExecutionAccountingRecord(
            method_id="legacy_cnn_residual",
            trace_id=observed.trace_id,
            cycle_index=1,
            action_valid_cycle=7,
            source_to_action_cycles=6,
            logical_deadline_miss=False,
            update_due=True,
            update_macs=int(cnn["macs_per_inference"]),
            private_model_state_bytes=int(cnn["private_model_state_bytes"]),
            transient_workspace_bytes=int(cnn["transient_workspace_bytes_lower_bound"]),
            host_update_wallclock_us=1.0,
            host_update_deadline_miss=False,
            board_measured_deadline_miss=None,
        )
        assert_accounting_conforms(record_value)

    record("cnn_matched_budget_overflow", cnn_budget)
    record(
        "oracle_in_deployable_accounting",
        lambda: assert_accounting_conforms(
            ExecutionAccountingRecord(
                method_id="hidden_state_oracle",
                trace_id=observed.trace_id,
                cycle_index=1,
                action_valid_cycle=7,
                source_to_action_cycles=6,
                logical_deadline_miss=False,
                update_due=False,
                update_macs=0,
                private_model_state_bytes=0,
                transient_workspace_bytes=0,
                host_update_wallclock_us=0.0,
                host_update_deadline_miss=False,
                board_measured_deadline_miss=None,
            )
        ),
    )
    return cases


def _adapter_bindings() -> list[dict[str, Any]]:
    wrapper = Path("cnn_fpga/benchmark/unified_comparator_runner.py")
    rows = (
        ("standard_binning", "paired_residuals_from_packets+fixed_central_rule", Path("cnn_fpga/benchmark/standard_binning_baseline.py"), "qualified_common_trace"),
        ("static_joint_map", "paired_residuals_from_packets+physics.ideal_gkp_decoder.map_decode_2d", Path("physics/ideal_gkp_decoder.py"), "qualified_software_joint_lane"),
        ("window_map", "LatestWindowPeriodicPredictor", Path("cnn_fpga/decoder/periodic_adaptive_map.py"), "qualified_common_trace"),
        ("ewma_adaptive_map", "PeriodicMomentEWMA", Path("cnn_fpga/decoder/periodic_adaptive_map.py"), "qualified_common_trace"),
        ("kalman_adaptive_map", "ConstantVelocityPeriodicKalman", Path("cnn_fpga/decoder/periodic_adaptive_map.py"), "qualified_common_trace"),
        ("legacy_cnn_residual", "_legacy_cnn_witness+predict_from_artifact", Path("cnn_fpga/model/tiny_cnn.py"), "executed_then_demoted_ablation_only"),
        ("proposed_route_a", "RouteAIntegrationAdapter.run_window", wrapper, "integration_only_pending_T6.6.2_T6.6.3"),
        ("hidden_state_oracle", "_oracle_run+physics.oracle_map.oracle_map_2d", Path("physics/oracle_map.py"), "isolated_privileged_upper_bound"),
    )
    return [
        {
            "method_id": method_id,
            "wrapper_path": wrapper.as_posix(),
            "wrapper_sha256": _sha256(wrapper),
            "algorithm_path": algorithm.as_posix(),
            "algorithm_sha256": _sha256(algorithm),
            "entrypoint": entrypoint,
            "qualification_status": status,
            "t6_5_2_manifest_mutated": False,
        }
        for method_id, entrypoint, algorithm, status in rows
    ]


def recompute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    deployable = report["deployable_common_trace_table"]
    ablation = report["ablation_table"]
    oracle = report["oracle_table"]
    expected = set(COMMON_TRACE_METHODS)
    method_ids = {row["method_id"] for row in deployable}
    input_hashes = {row["trace_sha256"] for row in deployable}
    accounting = [item for row in deployable for item in row["accounting"]]
    route_rows = next(row for row in deployable if row["method_id"] == "proposed_route_a")
    route_reasons: Counter[str] = Counter()
    for row in route_rows["window_rows"]:
        route_reasons.update(row["route_reason_counts"])
    prefix = report["prefix_causality_audit"]["rows"]
    cnn = ablation[0]
    metrics = report["qualification_metrics"]
    bindings = report["adapter_bindings"]
    return {
        "registry_exactly_seven_deployable_candidates_plus_oracle": (
            tuple(row.method_id for row in canonical_method_manifests()) == DEPLOYABLE_METHOD_IDS
            and oracle["manifest"]["method_id"] == "hidden_state_oracle"
        ),
        "versioned_adapter_overlay_binds_all_methods_without_mutating_t652": (
            {row["method_id"] for row in bindings}
            == set(DEPLOYABLE_METHOD_IDS) | {"hidden_state_oracle"}
            and all(
                len(row["wrapper_sha256"]) == 64
                and len(row["algorithm_sha256"]) == 64
                and row["t6_5_2_manifest_mutated"] is False
                for row in bindings
            )
        ),
        "six_schema_conforming_methods_execute_same_packet_trace": (
            method_ids == expected and len(input_hashes) == 1
        ),
        "all_common_trace_methods_have_real_decision_hashes": all(
            len(row["decision_sha256"]) == 64 and row["actual_execution"].startswith("real_")
            for row in deployable
        ),
        "packet_bridge_is_q_then_p_adjacent_and_wire_validated": bool(
            report["packet_bridge"]["all_windows_exact_pair_count"]
            and report["packet_bridge"]["wire_roundtrip_validated_during_materialization"]
        ),
        "parameter_window_and_regime_cadence_match_contract": (
            report["packet_bridge"]["scalar_packets_per_parameter_window"] == 2048
            and report["packet_bridge"]["pairs_per_parameter_window"] == 1024
            and report["packet_bridge"]["parameter_window_stride_cycles"] == 4000
            and report["packet_bridge"]["regime_window_cycles"] == 32
        ),
        "all_eligible_accounting_records_pass_exact_six_cycle_and_budget_contract": all(
            item["contract_passed"] for item in accounting
        ),
        "standard_rule_has_exhaustive_current_lut_action_equivalence": (
            report["standard_lut_equivalence"]["phase_code_cases"] == 2048
            and report["standard_lut_equivalence"]["mismatches"] == 0
        ),
        "periodic_feature_frontend_has_exhaustive_product_free_grid_equivalence": (
            report["periodic_feature_grid_equivalence"]["checked_complex_product_identities"]
            == 2 * 1024 * 1024
            and report["periodic_feature_grid_equivalence"]["max_absolute_complex_error"]
            <= report["periodic_feature_grid_equivalence"]["tolerance"]
        ),
        "joint_and_adaptive_software_rows_do_not_claim_current_rtl_equivalence": all(
            "current_rtl_bit_exact" not in row["current_rtl_status"]
            for row in deployable
            if row["method_id"] != "standard_binning"
        ),
        "prefix_causality_holds_for_every_common_trace_method": (
            {row["method_id"] for row in prefix} == expected
            and all(row["prefix_equal"] and row["future_input_changed"] for row in prefix)
        ),
        "route_a_executes_real_hmm_and_candidate_but_is_not_formal_claim": (
            route_rows["ranking_status"] == "integration_qualification_only_not_formal_result"
            and sum(row["hmm_updates"] for row in route_rows["window_rows"]) > 0
            and route_rows["actual_execution"].endswith("thresholds_not_locked")
            and route_reasons["normal_kalman_candidate"] > 0
            and sum(
                value for reason, value in route_reasons.items() if "trusted_static" in reason
            )
            > 0
        ),
        "legacy_cnn_checkpoint_really_executes_without_online_labels": (
            cnn["actual_checkpoint_loaded"]
            and len(cnn["witness_prediction_sha256"]) == 64
            and cnn["adapter_consumed_keys"] == ["histograms"]
            and not cnn["labels_or_target_consumed_online"]
            and cnn["all_three_assets_match_t5_4_3_anchor"]
        ),
        "legacy_cnn_is_automatically_demoted_on_schema_and_budget": (
            not cnn["input_schema_conforms"]
            and not cnn["matched_budget_conforms"]
            and cnn["ranking_status"].startswith("ablation_only")
            and cnn["common_trace_sha256"] is None
        ),
        "oracle_is_physically_separate_and_never_ranked_deployable": (
            oracle["physical_table"] == "oracle_upper_bound_only"
            and oracle["included_in_deployable_ranking"] is False
            and oracle["accounting"] is None
        ),
        "mutation_audit_fails_closed": (
            len(report["mutation_audit"]) >= 7
            and all(row["rejected"] for row in report["mutation_audit"])
        ),
        "qualification_metric_rows_are_complete_and_finite": (
            len(metrics["method_rows"]) == 7
            and len(metrics["window_rows"]) == 7 * report["config"]["windows"]
            and all(
                row["samples"] == report["config"]["windows"] * 1024
                and all(math.isfinite(float(row[field])) for field in ("logical_error_rate", "nll", "brier"))
                for row in metrics["method_rows"]
            )
        ),
        "qualification_metrics_are_explicitly_nonformal": (
            report["evidence_boundary"]["formal_t6_7_result"] is False
            and report["evidence_boundary"]["thresholds_locked"] is False
            and report["evidence_boundary"]["board_measured"] is False
        ),
    }


def build_report(config: RunnerConfig | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = RunnerConfig() if config is None else config
    observed, truth = materialize_qualification_trace(settings)
    runs = run_deployable_methods(observed, settings)
    oracle = _oracle_run(observed, truth)
    cnn = _legacy_cnn_witness(settings)
    method_metrics, metric_windows = _evaluate_runs(runs, oracle, truth)
    prefix = _prefix_causality(observed, runs, settings)
    deployable_rows = []
    for method_id in COMMON_TRACE_METHODS:
        run = runs[method_id]
        deployable_rows.append(
            {
                "method_id": method_id,
                "trace_sha256": run.trace_sha256,
                "decision_sha256": run.decision_sha256,
                "actual_execution": run.actual_execution,
                "current_rtl_status": run.current_rtl_status,
                "ranking_status": run.ranking_status,
                "private_model_state_bytes": run.private_model_state_bytes,
                "max_update_macs": run.max_update_macs,
                "max_transient_workspace_bytes": run.max_transient_workspace_bytes,
                "accounting": [
                    {**asdict(record), "contract_passed": True}
                    for record in run.accounting
                ],
                "window_rows": list(run.window_rows),
            }
        )
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract_sha256(),
        "config": asdict(settings),
        "trace": {
            "trace_id": observed.trace_id,
            "trace_sha256": observed.trace_sha256,
            "calibration_sha256": observed.calibration_sha256,
            "windows": len(observed.windows),
            "truth_trace_physically_separate": True,
        },
        "packet_bridge": {
            "wire_scalar_semantics": "one selected quadrature syndrome per fast cycle",
            "joint_sample_rule": "adjacent phase0_q then phase1_p packets only",
            "regime_cycle_rule": "new selected phase plus causal last-observed other phase",
            "scalar_packets_per_parameter_window": settings.scalar_packets_per_parameter_window,
            "pairs_per_parameter_window": settings.pairs_per_parameter_window,
            "parameter_window_stride_cycles": settings.parameter_window_stride_cycles,
            "regime_window_cycles": settings.regime_window_cycles,
            "all_windows_exact_pair_count": all(
                len(window.packets) == settings.scalar_packets_per_parameter_window
                for window in observed.windows
            ),
            "wire_roundtrip_validated_during_materialization": True,
            "floating_syndrome_side_channel": "prohibited",
        },
        "operation_accounting": {
            "mac_unit": "one real multiply-accumulate in the method-private update",
            "quantized_periodic_feature_frontend": (
                "10-bit phase LUT plus modular index sum/difference and integer accumulation; "
                "no transcendental or real MAC is required after exhaustive code-grid mapping"
            ),
            "host_runner_executes": "existing numpy complex-moment implementation and reports measured wallclock",
            "method_private_costs_are_conservative_static_counts": True,
            "shared_input_packet_buffer_and_common_MAP_LUT_excluded": True,
        },
        "adapter_bindings": _adapter_bindings(),
        "deployable_common_trace_table": deployable_rows,
        "ablation_table": [cnn],
        "oracle_table": {
            key: value
            for key, value in oracle.items()
            if key not in ("decisions", "posteriors")
        },
        "qualification_metrics": {
            "method_rows": method_metrics,
            "window_rows": metric_windows,
            "ranking_prohibited": True,
        },
        "standard_lut_equivalence": _standard_lut_equivalence(),
        "periodic_feature_grid_equivalence": _periodic_feature_grid_equivalence(),
        "prefix_causality_audit": prefix,
        "mutation_audit": _mutation_audit(observed, cnn),
        "evidence_boundary": {
            "formal_t6_7_result": False,
            "thresholds_locked": False,
            "route_policy_complete": False,
            "board_measured": False,
            "cnn_ler_comparable": False,
            "allowed_claim": (
                "the unified runner executes six common-trace observed-only methods, a real but "
                "pre-lock Route-A integration adapter, a physically isolated oracle, and a real "
                "legacy CNN witness that fails schema/budget promotion"
            ),
        },
        "source_bindings": [
            {"path": path.as_posix(), "sha256": _sha256(path)}
            for path in (
                HMM_CHECKPOINT,
                CNN_MODEL,
                CNN_MANIFEST,
                CNN_TEST_SPLIT,
                CNN_PARENT_REPORT,
                PARENT_MIXED_REPORT,
                Path("cnn_fpga/benchmark/unified_comparator_runner.py"),
                Path("cnn_fpga/runtime/unified_execution_contract.py"),
            )
        ],
    }
    report["gates"] = recompute_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
        "all_passed": all(report["gates"].values()),
    }
    report["verdict"] = (
        "PASS_UNIFIED_COMPARATOR_RUNNER_QUALIFIED"
        if report["gate_summary"]["all_passed"]
        else "FAIL_UNIFIED_COMPARATOR_RUNNER"
    )
    rows: list[dict[str, Any]] = []
    for row in method_metrics:
        rows.append({"row_type": "method_metric", **row})
    for row in metric_windows:
        rows.append({"row_type": "window_metric", **row})
    for row in deployable_rows:
        rows.append(
            {
                "row_type": "method_cost",
                "method_id": row["method_id"],
                "private_model_state_bytes": row["private_model_state_bytes"],
                "max_update_macs": row["max_update_macs"],
                "max_transient_workspace_bytes": row["max_transient_workspace_bytes"],
                "decision_sha256": row["decision_sha256"],
                "ranking_status": row["ranking_status"],
            }
        )
    rows.append(
        {
            "row_type": "method_cost",
            "method_id": "legacy_cnn_residual",
            "private_model_state_bytes": cnn["private_model_state_bytes"],
            "max_update_macs": cnn["macs_per_inference"],
            "max_transient_workspace_bytes": cnn["transient_workspace_bytes_lower_bound"],
            "decision_sha256": cnn["witness_prediction_sha256"],
            "ranking_status": cnn["ranking_status"],
        }
    )
    for name, passed in report["gates"].items():
        rows.append({"row_type": "gate", "gate": name, "passed": passed})
    return report, rows


def verify_report(report: Mapping[str, Any], *, verify_sources: bool = True) -> None:
    if report.get("task_id") != TASK_ID or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("T6.6.1 report identity mismatch")
    if report.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("T6.6.1 protocol mismatch")
    gates = recompute_gates(report)
    if gates != report.get("gates") or not all(gates.values()):
        raise ValueError("T6.6.1 gates are stale or failed")
    expected_summary = {
        "passed": len(gates),
        "total": len(gates),
        "all_passed": True,
    }
    if report.get("gate_summary") != expected_summary:
        raise ValueError("T6.6.1 gate summary mismatch")
    if report.get("verdict") != "PASS_UNIFIED_COMPARATOR_RUNNER_QUALIFIED":
        raise ValueError("T6.6.1 verdict mismatch")
    if verify_sources:
        for binding in report["source_bindings"]:
            if _sha256(binding["path"]) != binding["sha256"]:
                raise ValueError(f"source binding drifted: {binding['path']}")
        for binding in report["adapter_bindings"]:
            for prefix in ("wrapper", "algorithm"):
                if _sha256(binding[f"{prefix}_path"]) != binding[f"{prefix}_sha256"]:
                    raise ValueError(
                        f"adapter {prefix} binding drifted: {binding[f'{prefix}_path']}"
                    )


def write_report(
    artifact: str | Path = DEFAULT_ARTIFACT,
    source_data: str | Path = DEFAULT_SOURCE_DATA,
    config: RunnerConfig | None = None,
) -> dict[str, Any]:
    report, rows = build_report(config)
    verify_report(report)
    artifact_path = _repo_path(artifact)
    source_path = _repo_path(source_data)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--source-data", default=DEFAULT_SOURCE_DATA.as_posix())
    args = parser.parse_args(argv)
    report = write_report(args.artifact, args.source_data)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "gates": report["gate_summary"],
                "trace_sha256": report["trace"]["trace_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABlation_ONLY_METHODS",
    "COMMON_TRACE_METHODS",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "IsolatedTruthTrace",
    "MethodRun",
    "ObservedTrace",
    "ObservedTraceWindow",
    "PROTOCOL_ID",
    "RunnerConfig",
    "SCHEMA_VERSION",
    "TASK_ID",
    "build_report",
    "materialize_qualification_trace",
    "paired_residuals_from_packets",
    "recompute_gates",
    "run_deployable_methods",
    "verify_report",
    "write_report",
]
