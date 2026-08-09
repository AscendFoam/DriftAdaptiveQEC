"""sBs 位移故障注入与 syndrome-trend 复现。

该模块复现 Sivak 2023 Fig. 4(c) 的定性物理结构：位移到最近逻辑操作
（间隔 ``l_S/2``）的距离决定初始 error-hierarchy depth，因此 ``0`` 与
``l_S/2`` 附近的 syndrome trace 较短，而 ``l_S/4`` 最长。

数值 kernel 是显式、可复现的项目 modeling assumption，不是 Fig. 4(c) 数据
拟合或目标装置标定。trajectory 直接消费 T2.0.2 ``SBSErrorSpaceInstrument`` 的
transition matrices，并通过 T2.0.3 ``SBSObservationResetModel`` 的完整
preparation/readout/reset kernels 生成 observed syndrome。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._shared.validation import integer as _integer
from .sbs_error_space import (
    SBS_OUTCOMES,
    SBS_PROTOCOL_ID,
    SBSErrorSpaceInstrument,
    make_trickle_down_chain,
)
from .sbs_observation_reset import (
    HIDDEN_ANCILLA_STATES,
    IDEAL_ANCILLA_STATES,
    OBSERVED_CLASSES,
    SBSObservationResetModel,
    ideal_syndrome_from_kraus,
    make_persistent_leakage_model,
)
from ._shared.sampling import categorical_rows as _sample_rows


MODEL_SCOPE = "protocol_aligned_displacement_fault_trend_not_device_calibrated"
PRIMARY_SOURCE_PATH = (
    "relative_papers/Real-time_quantum_error_correction_beyond_break-even/"
    "Real-time_quantum_error_correction_beyond_break-even.md"
)
PRIMARY_SOURCE_ANCHORS = (
    {
        "line": 115,
        "fragment": "a displacement of amplitude $ l_{S}/4 $ makes a large-distance error",
        "role": "figure_caption",
    },
    {
        "line": 137,
        "fragment": "distance to the closest logical operation",
        "role": "nonmonotonic_trend",
    },
    {
        "line": 139,
        "fragment": "consecutive e outcomes in the same-quadrature cycles",
        "role": "same_quadrature_run",
    },
)


def _real_probability(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real probability")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real probability") from exc
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    return _integer(value, name, minimum)


def distance_to_closest_logical_operation(
    amplitude_over_lattice: ArrayLike,
    *,
    logical_spacing_over_lattice: float = 0.5,
) -> NDArray[np.float64]:
    """返回到最近逻辑位移的归一化距离，范围为 ``[0,1]``。

    对 square GKP，逻辑位移间隔是 ``l_S/2``。返回值用半个逻辑间隔
    ``l_S/4`` 归一化，因此 ``epsilon/l_S=0.25`` 返回 1，而 0 和 0.5 返回 0。
    函数对负位移和多个逻辑周期保持周期性。
    """

    values = np.asarray(amplitude_over_lattice, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("amplitude_over_lattice must contain finite values")
    if isinstance(logical_spacing_over_lattice, bool):
        raise TypeError("logical_spacing_over_lattice must be a positive real")
    spacing = float(logical_spacing_over_lattice)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("logical_spacing_over_lattice must be finite and positive")
    phase = np.mod(values, spacing)
    distance = np.minimum(phase, spacing - phase)
    normalized = distance / (0.5 * spacing)
    return np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float64)


@dataclass(frozen=True)
class DisplacementFaultSweepConfig:
    """预注册的 T2.0.5 sweep、统计 seed 与趋势容差。"""

    amplitudes_over_lattice: tuple[float, ...] = (
        0.0,
        0.0625,
        0.125,
        0.1875,
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.5,
    )
    shots: int = 4096
    cycles: int = 20
    seed: int = 2026071405
    bootstrap_seed: int = 2026071406
    bootstrap_replicates: int = 500
    confidence_level: float = 0.95
    max_recovery_depth: int = 6
    one_step_recovery_probability: float = 0.88
    fault_quadrature: str = "Z"
    false_e_given_g: float = 0.005
    e_detection_probability: float = 0.98
    expected_peak_amplitude: float = 0.25
    peak_location_tolerance: float = 0.0625
    minimum_midpoint_endpoint_run_margin: float = 2.0
    minimum_left_spearman: float = 0.95
    maximum_right_spearman: float = -0.95
    maximum_mirror_run_difference: float = 0.30
    maximum_endpoint_initial_depth: float = 0.05
    maximum_midpoint_depth_error: float = 0.05
    minimum_midpoint_early_late_e_margin: float = 0.25
    maximum_unaffected_e_probability: float = 0.06
    minimum_midpoint_recovered_fraction: float = 0.98
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        amplitudes = tuple(float(value) for value in self.amplitudes_over_lattice)
        if len(amplitudes) < 5 or not np.all(np.isfinite(amplitudes)):
            raise ValueError("amplitudes_over_lattice must contain at least five finite values")
        if any(not 0.0 <= value <= 0.5 for value in amplitudes):
            raise ValueError("the preregistered main-lobe amplitudes must lie in [0, 0.5]")
        if any(right <= left for left, right in zip(amplitudes, amplitudes[1:])):
            raise ValueError("amplitudes_over_lattice must be strictly increasing")
        for required in (0.0, 0.25, 0.5):
            if not any(np.isclose(value, required, rtol=0.0, atol=1.0e-12) for value in amplitudes):
                raise ValueError("amplitudes_over_lattice must include 0, 0.25 and 0.5")
        object.__setattr__(self, "amplitudes_over_lattice", amplitudes)
        object.__setattr__(self, "shots", _positive_integer(self.shots, "shots", minimum=64))
        object.__setattr__(self, "cycles", _positive_integer(self.cycles, "cycles", minimum=3))
        object.__setattr__(
            self,
            "bootstrap_replicates",
            _positive_integer(self.bootstrap_replicates, "bootstrap_replicates", minimum=100),
        )
        object.__setattr__(
            self,
            "max_recovery_depth",
            _positive_integer(self.max_recovery_depth, "max_recovery_depth", minimum=2),
        )
        if self.cycles < self.max_recovery_depth:
            raise ValueError("cycles must be at least max_recovery_depth")
        for name in ("seed", "bootstrap_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        for name in (
            "confidence_level",
            "one_step_recovery_probability",
            "false_e_given_g",
            "e_detection_probability",
            "minimum_midpoint_recovered_fraction",
        ):
            object.__setattr__(self, name, _real_probability(getattr(self, name), name))
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly in (0, 1)")
        if self.one_step_recovery_probability <= 0.0:
            raise ValueError("one_step_recovery_probability must be positive")
        if self.fault_quadrature not in {"X", "Z"}:
            raise ValueError("fault_quadrature must be X or Z")
        for name in (
            "expected_peak_amplitude",
            "peak_location_tolerance",
            "minimum_midpoint_endpoint_run_margin",
            "minimum_left_spearman",
            "maximum_right_spearman",
            "maximum_mirror_run_difference",
            "maximum_endpoint_initial_depth",
            "maximum_midpoint_depth_error",
            "minimum_midpoint_early_late_e_margin",
            "maximum_unaffected_e_probability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.peak_location_tolerance < 0.0:
            raise ValueError("peak_location_tolerance must be non-negative")
        if self.minimum_midpoint_endpoint_run_margin < 0.0:
            raise ValueError("minimum_midpoint_endpoint_run_margin must be non-negative")
        if not -1.0 <= self.minimum_left_spearman <= 1.0:
            raise ValueError("minimum_left_spearman must lie in [-1, 1]")
        if not -1.0 <= self.maximum_right_spearman <= 1.0:
            raise ValueError("maximum_right_spearman must lie in [-1, 1]")
        for name in (
            "maximum_mirror_run_difference",
            "maximum_endpoint_initial_depth",
            "maximum_midpoint_depth_error",
            "minimum_midpoint_early_late_e_margin",
            "maximum_unaffected_e_probability",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")


@dataclass(frozen=True)
class MeanEstimate:
    mean: float
    ci_low: float
    ci_high: float
    method: str
    confidence_level: float
    replicates: int


@dataclass(frozen=True)
class DisplacementTrendPoint:
    amplitude_over_lattice: float
    logical_distance: float
    initial_recovery_depth: MeanEstimate
    ideal_same_quadrature_max_e_run: MeanEstimate
    observed_same_quadrature_max_e_run: MeanEstimate
    restricted_recovery_cycles: MeanEstimate
    recovered_fraction_by_horizon: float
    affected_e_probability_by_cycle: tuple[float, ...]
    unaffected_e_probability_by_cycle: tuple[float, ...]
    censored_shots: int


@dataclass(frozen=True)
class TrendCheck:
    check_id: str
    passed: bool
    observed: float
    criterion: str
    limit: float
    detail: str


@dataclass(frozen=True)
class DisplacementTrendGate:
    passed: bool
    checks: tuple[TrendCheck, ...]

    @property
    def failed_check_ids(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.checks if not check.passed)


@dataclass(frozen=True)
class DisplacementFaultSweepResult:
    config: DisplacementFaultSweepConfig
    points: tuple[DisplacementTrendPoint, ...]
    gate: DisplacementTrendGate
    source_path: str = PRIMARY_SOURCE_PATH
    source_anchors: tuple[dict[str, object], ...] = PRIMARY_SOURCE_ANCHORS
    protocol_id: str = SBS_PROTOCOL_ID
    evidence_scope: str = MODEL_SCOPE
    device_calibrated: bool = False
    experimental_data_digitized_or_fitted: bool = False

    def require_pass(self) -> None:
        if not self.gate.passed:
            raise RuntimeError(
                "displacement fault trend gate failed: " + ", ".join(self.gate.failed_check_ids)
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["limitations"] = [
            "coarse-grained error-space depth, not a Fock-space displacement simulation",
            "readout and recovery probabilities are explicit project assumptions, not device calibration",
            "reproduces the qualitative nonmonotonic trend, not Fig. 4(c) pixel values",
            "no target-board timing, microwave waveform or real quantum-hardware evidence",
        ]
        return payload


@dataclass(frozen=True)
class _RawAmplitudeResult:
    initial_depth: NDArray[np.int64]
    ideal_max_run: NDArray[np.int64]
    observed_max_run: NDArray[np.int64]
    restricted_recovery_cycles: NDArray[np.int64]
    recovered: NDArray[np.bool_]
    affected_e: NDArray[np.bool_]
    unaffected_e: NDArray[np.bool_]


def _make_observation_model(config: DisplacementFaultSweepConfig) -> SBSObservationResetModel:
    readout_confusion = np.array(
        [
            [1.0 - config.false_e_given_g, config.false_e_given_g, 0.0],
            [1.0 - config.e_detection_probability, config.e_detection_probability, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return make_persistent_leakage_model(
        readout_confusion=readout_confusion,
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=0.0,
        counter_max=max(31, config.cycles),
        readout_provenance="T2.0.5 explicit sensitivity assumption; not device calibrated",
        parameter_provenance="T2.0.5 displacement-only isolation; leakage injection disabled",
    )


def _make_recovery_instrument(config: DisplacementFaultSweepConfig) -> SBSErrorSpaceInstrument:
    return make_trickle_down_chain(
        max_depth=config.max_recovery_depth,
        one_step_probability=config.one_step_recovery_probability,
        two_step_probability=0.0,
        ge_fraction=1.0 if config.fault_quadrature == "X" else 0.0,
    )


def _max_consecutive_true(values: NDArray[np.bool_]) -> NDArray[np.int64]:
    if values.ndim != 2:
        raise ValueError("values must be a shot-by-cycle boolean matrix")
    current = np.zeros(values.shape[0], dtype=np.int64)
    maximum = np.zeros(values.shape[0], dtype=np.int64)
    for cycle in range(values.shape[1]):
        current = np.where(values[:, cycle], current + 1, 0)
        maximum = np.maximum(maximum, current)
    return maximum


def _simulate_amplitude(
    amplitude: float,
    *,
    config: DisplacementFaultSweepConfig,
    instrument: SBSErrorSpaceInstrument,
    observation_model: SBSObservationResetModel,
    rng: np.random.Generator,
) -> _RawAmplitudeResult:
    severity = float(distance_to_closest_logical_operation(amplitude))
    initial_depth = rng.binomial(
        config.max_recovery_depth,
        severity,
        size=config.shots,
    ).astype(np.int64)
    depth = initial_depth.copy()
    depth_history = np.empty((config.shots, config.cycles + 1), dtype=np.int64)
    depth_history[:, 0] = depth
    ideal_outcome = np.empty((config.shots, config.cycles), dtype=np.int64)

    transition_by_source: list[NDArray[np.float64]] = []
    size = len(instrument.subspaces)
    for source in range(size):
        flat = np.concatenate(
            [instrument.transition_probabilities[outcome][:, source] for outcome in SBS_OUTCOMES]
        )
        transition_by_source.append(flat[None, :])

    for cycle in range(config.cycles):
        target_depth = np.empty_like(depth)
        outcome_index = np.empty_like(depth)
        for source in range(size):
            mask = depth == source
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            rows = np.repeat(transition_by_source[source], count, axis=0)
            choice = _sample_rows(rows, rng)
            outcome_index[mask] = choice // size
            target_depth[mask] = choice % size
        depth = target_depth
        ideal_outcome[:, cycle] = outcome_index
        depth_history[:, cycle + 1] = depth

    ideal_pairs = np.array(
        [ideal_syndrome_from_kraus(label).as_tuple() for label in SBS_OUTCOMES],
        dtype=object,
    )
    ideal_state_index = np.empty((config.shots, config.cycles, 2), dtype=np.int64)
    for constituent_index in range(2):
        ideal_state_index[:, :, constituent_index] = np.vectorize(
            IDEAL_ANCILLA_STATES.index
        )(ideal_pairs[ideal_outcome, constituent_index])

    carry = np.zeros(config.shots, dtype=np.int64)
    observed_index = np.empty((config.shots, config.cycles, 2), dtype=np.int64)
    for cycle in range(config.cycles):
        for constituent in range(2):
            ideal = ideal_state_index[:, cycle, constituent]
            preparation_rows = observation_model.preparation_kernel[carry, ideal]
            hidden_pre = _sample_rows(preparation_rows, rng)
            observed = _sample_rows(observation_model.readout_confusion[hidden_pre], rng)
            reset_rows = observation_model.reset_kernel[observed, hidden_pre]
            carry = _sample_rows(reset_rows, rng)
            observed_index[:, cycle, constituent] = observed

    affected_constituent = 0 if config.fault_quadrature == "X" else 1
    unaffected_constituent = 1 - affected_constituent
    e_index = OBSERVED_CLASSES.index("e")
    ideal_e_index = IDEAL_ANCILLA_STATES.index("e")
    ideal_affected_e = ideal_state_index[:, :, affected_constituent] == ideal_e_index
    observed_affected_e = observed_index[:, :, affected_constituent] == e_index
    observed_unaffected_e = observed_index[:, :, unaffected_constituent] == e_index

    zero_after_start = depth_history[:, 1:] == 0
    recovered = np.any(zero_after_start, axis=1) | (initial_depth == 0)
    first_zero = np.argmax(zero_after_start, axis=1) + 1
    restricted_recovery = np.where(
        initial_depth == 0,
        0,
        np.where(np.any(zero_after_start, axis=1), first_zero, config.cycles + 1),
    ).astype(np.int64)

    return _RawAmplitudeResult(
        initial_depth=initial_depth,
        ideal_max_run=_max_consecutive_true(ideal_affected_e),
        observed_max_run=_max_consecutive_true(observed_affected_e),
        restricted_recovery_cycles=restricted_recovery,
        recovered=recovered,
        affected_e=observed_affected_e,
        unaffected_e=observed_unaffected_e,
    )


def _bootstrap_mean(
    values: ArrayLike,
    *,
    rng: np.random.Generator,
    config: DisplacementFaultSweepConfig,
) -> MeanEstimate:
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (config.shots,) or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be one finite value per shot")
    bootstrap_means = np.empty(config.bootstrap_replicates, dtype=np.float64)
    batch_size = min(50, config.bootstrap_replicates)
    completed = 0
    while completed < config.bootstrap_replicates:
        batch = min(batch_size, config.bootstrap_replicates - completed)
        indices = rng.integers(0, config.shots, size=(batch, config.shots))
        bootstrap_means[completed : completed + batch] = np.mean(data[indices], axis=1)
        completed += batch
    tail = 0.5 * (1.0 - config.confidence_level)
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return MeanEstimate(
        mean=float(np.mean(data)),
        ci_low=float(low),
        ci_high=float(high),
        method="percentile_nonparametric_bootstrap",
        confidence_level=config.confidence_level,
        replicates=config.bootstrap_replicates,
    )


def _rankdata(values: Sequence[float]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman inputs must have the same length of at least two")
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _evaluate_gate(
    points: tuple[DisplacementTrendPoint, ...],
    config: DisplacementFaultSweepConfig,
) -> DisplacementTrendGate:
    amplitudes = np.asarray([point.amplitude_over_lattice for point in points])
    runs = np.asarray([point.observed_same_quadrature_max_e_run.mean for point in points])
    midpoint_index = int(np.flatnonzero(np.isclose(amplitudes, 0.25, atol=1.0e-12))[0])
    left = slice(0, midpoint_index + 1)
    right = slice(midpoint_index, amplitudes.size)
    endpoint_indices = (0, amplitudes.size - 1)
    midpoint = points[midpoint_index]

    peak_amplitude = float(amplitudes[int(np.argmax(runs))])
    midpoint_endpoint_margin = float(
        midpoint.observed_same_quadrature_max_e_run.ci_low
        - max(points[index].observed_same_quadrature_max_e_run.ci_high for index in endpoint_indices)
    )
    left_spearman = _spearman(amplitudes[left], runs[left])
    right_spearman = _spearman(amplitudes[right], runs[right])
    mirror_difference = float(
        max(abs(runs[index] - runs[-1 - index]) for index in range(amplitudes.size // 2))
    )
    endpoint_depth = float(
        max(points[index].initial_recovery_depth.mean for index in endpoint_indices)
    )
    midpoint_depth_error = abs(
        midpoint.initial_recovery_depth.mean - config.max_recovery_depth
    )
    early_late_margin = float(
        np.mean(midpoint.affected_e_probability_by_cycle[:3])
        - np.mean(midpoint.affected_e_probability_by_cycle[-3:])
    )
    max_unaffected = float(
        max(max(point.unaffected_e_probability_by_cycle) for point in points)
    )

    checks = (
        TrendCheck(
            "peak_near_lS_over_4",
            abs(peak_amplitude - config.expected_peak_amplitude)
            <= config.peak_location_tolerance,
            peak_amplitude,
            "abs(observed-expected) <= tolerance",
            config.peak_location_tolerance,
            f"expected={config.expected_peak_amplitude}",
        ),
        TrendCheck(
            "midpoint_run_separated_from_endpoints",
            midpoint_endpoint_margin >= config.minimum_midpoint_endpoint_run_margin,
            midpoint_endpoint_margin,
            "midpoint CI-low - max(endpoint CI-high) >= limit",
            config.minimum_midpoint_endpoint_run_margin,
            "bootstrap interval separation",
        ),
        TrendCheck(
            "left_branch_monotone_increasing",
            left_spearman >= config.minimum_left_spearman,
            left_spearman,
            "Spearman rho >= limit",
            config.minimum_left_spearman,
            "0 to lS/4",
        ),
        TrendCheck(
            "right_branch_monotone_decreasing",
            right_spearman <= config.maximum_right_spearman,
            right_spearman,
            "Spearman rho <= limit",
            config.maximum_right_spearman,
            "lS/4 to lS/2",
        ),
        TrendCheck(
            "mirror_symmetry",
            mirror_difference <= config.maximum_mirror_run_difference,
            mirror_difference,
            "max paired absolute run difference <= limit",
            config.maximum_mirror_run_difference,
            "pairs epsilon and lS/2-epsilon",
        ),
        TrendCheck(
            "endpoint_depth_near_zero",
            endpoint_depth <= config.maximum_endpoint_initial_depth,
            endpoint_depth,
            "max endpoint mean depth <= limit",
            config.maximum_endpoint_initial_depth,
            "logical identity and logical flip endpoints",
        ),
        TrendCheck(
            "midpoint_depth_near_maximum",
            midpoint_depth_error <= config.maximum_midpoint_depth_error,
            midpoint_depth_error,
            "abs(mean depth-max depth) <= limit",
            config.maximum_midpoint_depth_error,
            "large-distance midpoint",
        ),
        TrendCheck(
            "midpoint_syndrome_trace_decays",
            early_late_margin >= config.minimum_midpoint_early_late_e_margin,
            early_late_margin,
            "mean P_e(cycles 1:3)-mean P_e(last 3) >= limit",
            config.minimum_midpoint_early_late_e_margin,
            "affected quadrature",
        ),
        TrendCheck(
            "unaffected_quadrature_negative_control",
            max_unaffected <= config.maximum_unaffected_e_probability,
            max_unaffected,
            "max P_e in unaffected quadrature <= limit",
            config.maximum_unaffected_e_probability,
            "detects quadrature mixing/reset carry contamination",
        ),
        TrendCheck(
            "midpoint_recovers_within_horizon",
            midpoint.recovered_fraction_by_horizon
            >= config.minimum_midpoint_recovered_fraction,
            midpoint.recovered_fraction_by_horizon,
            "recovered fraction >= limit",
            config.minimum_midpoint_recovered_fraction,
            f"horizon={config.cycles} full cycles",
        ),
    )
    return DisplacementTrendGate(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def run_displacement_fault_sweep(
    config: DisplacementFaultSweepConfig | None = None,
) -> DisplacementFaultSweepResult:
    """运行预注册 sweep 并返回带 bootstrap CI 和 failure diagnostics 的结果。"""

    actual = DisplacementFaultSweepConfig() if config is None else config
    if not isinstance(actual, DisplacementFaultSweepConfig):
        raise TypeError("config must be a DisplacementFaultSweepConfig or None")
    instrument = _make_recovery_instrument(actual)
    observation_model = _make_observation_model(actual)
    simulation_rng = np.random.default_rng(actual.seed)
    bootstrap_rng = np.random.default_rng(actual.bootstrap_seed)
    points: list[DisplacementTrendPoint] = []

    for amplitude in actual.amplitudes_over_lattice:
        raw = _simulate_amplitude(
            amplitude,
            config=actual,
            instrument=instrument,
            observation_model=observation_model,
            rng=simulation_rng,
        )
        points.append(
            DisplacementTrendPoint(
                amplitude_over_lattice=amplitude,
                logical_distance=float(distance_to_closest_logical_operation(amplitude)),
                initial_recovery_depth=_bootstrap_mean(
                    raw.initial_depth, rng=bootstrap_rng, config=actual
                ),
                ideal_same_quadrature_max_e_run=_bootstrap_mean(
                    raw.ideal_max_run, rng=bootstrap_rng, config=actual
                ),
                observed_same_quadrature_max_e_run=_bootstrap_mean(
                    raw.observed_max_run, rng=bootstrap_rng, config=actual
                ),
                restricted_recovery_cycles=_bootstrap_mean(
                    raw.restricted_recovery_cycles, rng=bootstrap_rng, config=actual
                ),
                recovered_fraction_by_horizon=float(np.mean(raw.recovered)),
                affected_e_probability_by_cycle=tuple(
                    float(value) for value in np.mean(raw.affected_e, axis=0)
                ),
                unaffected_e_probability_by_cycle=tuple(
                    float(value) for value in np.mean(raw.unaffected_e, axis=0)
                ),
                censored_shots=int(np.count_nonzero(~raw.recovered)),
            )
        )

    point_tuple = tuple(points)
    return DisplacementFaultSweepResult(
        config=actual,
        points=point_tuple,
        gate=_evaluate_gate(point_tuple, actual),
    )


def write_displacement_fault_report(
    result: DisplacementFaultSweepResult,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    """写出机器可读完整报告和紧凑趋势表。"""

    if not isinstance(result, DisplacementFaultSweepResult):
        raise TypeError("result must be a DisplacementFaultSweepResult")
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "amplitude_over_lattice",
                "logical_distance",
                "mean_initial_recovery_depth",
                "mean_ideal_same_quadrature_max_e_run",
                "mean_observed_same_quadrature_max_e_run",
                "observed_run_ci_low",
                "observed_run_ci_high",
                "restricted_mean_recovery_cycles",
                "recovered_fraction_by_horizon",
                "censored_shots",
            ),
        )
        writer.writeheader()
        for point in result.points:
            writer.writerow(
                {
                    "amplitude_over_lattice": point.amplitude_over_lattice,
                    "logical_distance": point.logical_distance,
                    "mean_initial_recovery_depth": point.initial_recovery_depth.mean,
                    "mean_ideal_same_quadrature_max_e_run": (
                        point.ideal_same_quadrature_max_e_run.mean
                    ),
                    "mean_observed_same_quadrature_max_e_run": (
                        point.observed_same_quadrature_max_e_run.mean
                    ),
                    "observed_run_ci_low": point.observed_same_quadrature_max_e_run.ci_low,
                    "observed_run_ci_high": point.observed_same_quadrature_max_e_run.ci_high,
                    "restricted_mean_recovery_cycles": point.restricted_recovery_cycles.mean,
                    "recovered_fraction_by_horizon": point.recovered_fraction_by_horizon,
                    "censored_shots": point.censored_shots,
                }
            )
