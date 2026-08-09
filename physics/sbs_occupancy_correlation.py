"""sBs syndrome-only occupancy 与 leakage-correlation 交叉验证。

该模块实现 Sivak 2023 S4E/S4F 的两个独立诊断：

1. 仅从 observed ``gg/gg/...`` string probability 拟合 ``a, lambda``，使用
   ``p_err ~= 1-lambda`` 与 ``<Pi0> ~= a*lambda`` 估计 code-space occupancy；
2. 对 observed non-g activity 计算长 lag correlation，并按 observed leakage run
   ``>=2`` 做整条 trajectory post-selection，检验长尾是否收缩。

hidden code/error/leakage state 只用于独立 truth estimate 和验证，绝不传入
syndrome-only estimator。transition/readout/leakage 参数均带明确 evidence scope；
这不是装置原始数据复现或 Fock-space simulator。
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
from .sbs_error_space import SBS_PROTOCOL_ID
from .sbs_observation_reset import (
    HIDDEN_ANCILLA_STATES,
    IDEAL_ANCILLA_STATES,
    OBSERVED_CLASSES,
    SBSObservationResetModel,
    make_persistent_leakage_model,
)
from ._shared.sampling import categorical_rows as _sample_rows


MODEL_SCOPE = "protocol_aligned_occupancy_correlation_effective_model_not_device_calibrated"
PRIMARY_SOURCE_PATH = (
    "relative_papers/Real-time_quantum_error_correction_beyond_break-even/"
    "Real-time_quantum_error_correction_beyond_break-even.md"
)
PRIMARY_SOURCE_ANCHORS = (
    {
        "line": 181,
        "fragment": "expectation value of the code projector $ \\langle \\Pi_{0}\\rangle=0.825\\pm 0.003 $",
        "role": "independent_hidden_state_reference",
    },
    {
        "line": 965,
        "fragment": "P([gg]^{n})=a\\lambda^{n}",
        "role": "syndrome_string_fit",
    },
    {
        "line": 967,
        "fragment": "p_{\\mathrm{err}} \\approx 1-\\lambda",
        "role": "occupancy_estimator_identity",
    },
    {
        "line": 1015,
        "fragment": "decay constant of 17.2 cycles",
        "role": "higher_leakage_duration",
    },
    {
        "line": 1027,
        "fragment": "p_{l,\\geq2}=(1.280\\pm0.002)\\times10^{-4}",
        "role": "long_leakage_rate",
    },
    {
        "line": 1039,
        "fragment": "length-two or longer leakage events",
        "role": "correlation_postselection",
    },
)


def _probability(value: float, name: str, *, strict: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real probability")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real probability") from exc
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if strict and not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly in (0, 1)")
    return number


@dataclass(frozen=True)
class OccupancyCorrelationConfig:
    """T2.0.6 生产规模、source-inspired 参数与 acceptance contract。"""

    shots: int = 600
    cycles: int = 1200
    burn_in_cycles: int = 200
    seed: int = 2026071407
    bootstrap_seed: int = 2026071408
    bootstrap_replicates: int = 400
    confidence_level: float = 0.95
    physical_error_probability: float = 0.13
    target_no_leakage_occupancy: float = 0.82
    single_cycle_leakage_rate: float = 5.48e-4
    higher_leakage_rate: float = 1.28e-4
    higher_leakage_mean_duration_cycles: float = 17.2
    readout_fidelity_g: float = 0.9997
    readout_fidelity_e: float = 0.9914
    max_recovery_depth: int = 64
    all_gg_string_lengths: tuple[int, ...] = tuple(range(2, 13))
    tail_lags: tuple[int, ...] = tuple(range(40, 201, 20))
    reference_occupancy: float = 0.82
    maximum_reference_occupancy_error: float = 0.02
    maximum_truth_syndrome_difference: float = 0.02
    maximum_physical_error_estimate_error: float = 0.02
    minimum_all_gg_fit_r_squared: float = 0.999
    minimum_retained_fraction: float = 0.75
    maximum_retained_fraction: float = 0.95
    minimum_pre_removal_tail_correlation: float = 0.001
    minimum_tail_difference_ci_low: float = 3.0e-4
    minimum_tail_shrink_ratio: float = 2.0
    maximum_post_removal_tail_correlation: float = 0.0015
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "shots", _integer(self.shots, "shots", 100))
        object.__setattr__(self, "cycles", _integer(self.cycles, "cycles", 100))
        object.__setattr__(
            self, "burn_in_cycles", _integer(self.burn_in_cycles, "burn_in_cycles", 0)
        )
        object.__setattr__(
            self,
            "bootstrap_replicates",
            _integer(self.bootstrap_replicates, "bootstrap_replicates", 100),
        )
        object.__setattr__(
            self, "max_recovery_depth", _integer(self.max_recovery_depth, "max_recovery_depth", 2)
        )
        for name in ("seed", "bootstrap_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        for name in (
            "confidence_level",
            "physical_error_probability",
            "target_no_leakage_occupancy",
            "readout_fidelity_g",
            "readout_fidelity_e",
            "reference_occupancy",
            "minimum_retained_fraction",
            "maximum_retained_fraction",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name, strict=True))
        for name in ("single_cycle_leakage_rate", "higher_leakage_rate"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.single_cycle_leakage_rate + self.higher_leakage_rate >= 1.0:
            raise ValueError("total leakage start probability must be below 1")
        duration = float(self.higher_leakage_mean_duration_cycles)
        if not np.isfinite(duration) or duration <= 2.0:
            raise ValueError("higher_leakage_mean_duration_cycles must exceed 2")
        object.__setattr__(self, "higher_leakage_mean_duration_cycles", duration)
        recovery_probability = self.recovery_probability
        if not 0.0 < recovery_probability <= 1.0:
            raise ValueError("target occupancy and physical error imply invalid recovery probability")
        lengths = tuple(int(value) for value in self.all_gg_string_lengths)
        if len(lengths) < 4 or any(value < 1 or value >= self.cycles for value in lengths):
            raise ValueError("all_gg_string_lengths need at least four values within the horizon")
        if any(right <= left for left, right in zip(lengths, lengths[1:])):
            raise ValueError("all_gg_string_lengths must be strictly increasing")
        object.__setattr__(self, "all_gg_string_lengths", lengths)
        lags = tuple(int(value) for value in self.tail_lags)
        if len(lags) < 4 or any(value < 2 or value >= self.cycles for value in lags):
            raise ValueError("tail_lags need at least four values within the horizon")
        if any(right <= left for left, right in zip(lags, lags[1:])):
            raise ValueError("tail_lags must be strictly increasing")
        object.__setattr__(self, "tail_lags", lags)
        for name in (
            "maximum_reference_occupancy_error",
            "maximum_truth_syndrome_difference",
            "maximum_physical_error_estimate_error",
            "minimum_all_gg_fit_r_squared",
            "minimum_pre_removal_tail_correlation",
            "minimum_tail_difference_ci_low",
            "minimum_tail_shrink_ratio",
            "maximum_post_removal_tail_correlation",
        ):
            number = float(getattr(self, name))
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, number)
        if self.minimum_retained_fraction >= self.maximum_retained_fraction:
            raise ValueError("minimum_retained_fraction must be below maximum_retained_fraction")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")

    @property
    def recovery_probability(self) -> float:
        return (
            self.target_no_leakage_occupancy
            * self.physical_error_probability
            / (1.0 - self.target_no_leakage_occupancy)
        )


@dataclass(frozen=True)
class OccupancyDataset:
    """truth 与 observed arrays；估计函数只接收 observed ``all_gg``。"""

    hidden_code_occupied: NDArray[np.bool_]
    hidden_error_depth: NDArray[np.int16]
    hidden_leakage_kind: NDArray[np.int8]
    observed_x: NDArray[np.int8]
    observed_z: NDArray[np.int8]
    all_gg: NDArray[np.bool_]
    non_g_activity: NDArray[np.bool_]
    observed_leakage: NDArray[np.bool_]
    seed: int
    truth_scope: str = "simulator_hidden_truth_not_deployable_input"
    observation_scope: str = "observed_syndrome_only"


@dataclass(frozen=True)
class IntervalEstimate:
    mean: float
    ci_low: float
    ci_high: float
    confidence_level: float
    method: str
    replicates: int


@dataclass(frozen=True)
class SyndromeOccupancyEstimate:
    string_lengths: tuple[int, ...]
    all_gg_probabilities: tuple[float, ...]
    fitted_a: float
    fitted_lambda: float
    physical_error_probability: float
    occupancy: float
    occupancy_statistical_ci: tuple[float, float]
    first_order_model_error_bound: float
    occupancy_combined_ci: tuple[float, float]
    r_squared_log_probability: float
    estimator_inputs: tuple[str, ...] = ("observed_all_gg_boolean_matrix",)


@dataclass(frozen=True)
class TailCorrelationEstimate:
    lags: tuple[int, ...]
    before_removal: tuple[float, ...]
    after_removal: tuple[float, ...]
    mean_before: IntervalEstimate
    mean_after: IntervalEstimate
    paired_difference: IntervalEstimate
    shrink_ratio: float
    retained_shots: int
    retained_fraction: float
    removal_rule: str = "drop full trajectories with observed leakage run >=2 cycles"


@dataclass(frozen=True)
class OccupancyCorrelationCheck:
    check_id: str
    passed: bool
    observed: float
    criterion: str
    limit: float
    detail: str


@dataclass(frozen=True)
class OccupancyCorrelationGate:
    passed: bool
    checks: tuple[OccupancyCorrelationCheck, ...]

    @property
    def failed_check_ids(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.checks if not check.passed)


@dataclass(frozen=True)
class OccupancyCorrelationResult:
    config: OccupancyCorrelationConfig
    hidden_occupancy: IntervalEstimate
    syndrome_occupancy: SyndromeOccupancyEstimate
    tail_correlation: TailCorrelationEstimate
    gate: OccupancyCorrelationGate
    source_path: str = PRIMARY_SOURCE_PATH
    source_anchors: tuple[dict[str, object], ...] = PRIMARY_SOURCE_ANCHORS
    protocol_id: str = SBS_PROTOCOL_ID
    evidence_scope: str = MODEL_SCOPE
    device_calibrated: bool = False
    experimental_data_reproduced: bool = False

    def require_pass(self) -> None:
        if not self.gate.passed:
            raise RuntimeError(
                "occupancy/correlation gate failed: " + ", ".join(self.gate.failed_check_ids)
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["limitations"] = [
            "hidden state is a coarse-grained recovery-depth process, not a Fock-space state",
            "leakage start/duration use source-inspired effective assumptions",
            "syndrome estimator is first-order and reports the explicit p_err squared model bound",
            "correlation removal is post-selection, not an online leakage-removal controller",
            "no target-device calibration or experimental raw-data fit",
        ]
        return payload


def _make_observation_model(config: OccupancyCorrelationConfig) -> SBSObservationResetModel:
    confusion = np.array(
        [
            [config.readout_fidelity_g, 1.0 - config.readout_fidelity_g, 0.0],
            [1.0 - config.readout_fidelity_e, config.readout_fidelity_e, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return make_persistent_leakage_model(
        readout_confusion=confusion,
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=0.0,
        counter_max=max(31, config.cycles),
        readout_provenance=(
            "Sivak S2C diagonal Fg/Fe; f/higher exact leakage classification is a T2.0.6 assumption"
        ),
        parameter_provenance=(
            "T2.0.6 external f/higher event process; higher spontaneous decay not controller reset"
        ),
    )


def simulate_occupancy_correlation_dataset(
    config: OccupancyCorrelationConfig | None = None,
) -> OccupancyDataset:
    """模拟同一批 shared trajectories，并结构化分开 truth/observed lanes。"""

    actual = OccupancyCorrelationConfig() if config is None else config
    if not isinstance(actual, OccupancyCorrelationConfig):
        raise TypeError("config must be an OccupancyCorrelationConfig or None")
    rng = np.random.default_rng(actual.seed)
    observation_model = _make_observation_model(actual)
    total_cycles = actual.burn_in_cycles + actual.cycles
    depth = np.zeros(actual.shots, dtype=np.int64)
    leakage_remaining = np.zeros(actual.shots, dtype=np.int64)
    leakage_kind = np.zeros(actual.shots, dtype=np.int8)  # 0 none, 1 f, 2 higher
    carry = np.zeros(actual.shots, dtype=np.int64)

    hidden_code = np.empty((actual.shots, actual.cycles), dtype=np.bool_)
    hidden_depth = np.empty((actual.shots, actual.cycles), dtype=np.int16)
    hidden_leakage = np.empty((actual.shots, actual.cycles), dtype=np.int8)
    observed_x = np.empty((actual.shots, actual.cycles), dtype=np.int8)
    observed_z = np.empty((actual.shots, actual.cycles), dtype=np.int8)

    g_ideal_index = IDEAL_ANCILLA_STATES.index("g")
    e_ideal_index = IDEAL_ANCILLA_STATES.index("e")
    leakage_observed_index = OBSERVED_CLASSES.index("leakage")
    f_hidden_index = HIDDEN_ANCILLA_STATES.index("f")
    higher_hidden_index = HIDDEN_ANCILLA_STATES.index("higher")
    g_hidden_index = HIDDEN_ANCILLA_STATES.index("g")
    geometric_probability = 1.0 / (actual.higher_leakage_mean_duration_cycles - 1.0)

    for absolute_cycle in range(total_cycles):
        inactive = leakage_remaining == 0
        starts = rng.random(actual.shots)
        start_higher = inactive & (starts < actual.higher_leakage_rate)
        start_f = inactive & (starts >= actual.higher_leakage_rate) & (
            starts < actual.higher_leakage_rate + actual.single_cycle_leakage_rate
        )
        durations = 1 + rng.geometric(geometric_probability, size=actual.shots)
        leakage_remaining[start_higher] = durations[start_higher]
        leakage_kind[start_higher] = 2
        leakage_remaining[start_f] = 1
        leakage_kind[start_f] = 1

        active_leakage = leakage_remaining > 0
        depth_at_cycle_start = depth.copy()
        code_now = (depth == 0) & ~active_leakage
        error_now = (depth > 0) & ~active_leakage

        # Inactive ancilla does not stabilize the code; each inactive cycle adds one
        # coarse-grained recovery level. This is explicit model scope, not a Fock claim.
        depth[active_leakage] = np.minimum(
            actual.max_recovery_depth, depth[active_leakage] + 1
        )
        recovers = error_now & (rng.random(actual.shots) < actual.recovery_probability)
        depth[recovers] -= 1
        new_error = code_now & (rng.random(actual.shots) < actual.physical_error_probability)
        depth[new_error] = 1

        x_cycle = np.empty(actual.shots, dtype=np.int64)
        z_cycle = np.empty(actual.shots, dtype=np.int64)
        x_cycle[active_leakage] = leakage_observed_index
        z_cycle[active_leakage] = leakage_observed_index
        carry[active_leakage & (leakage_kind == 1)] = f_hidden_index
        carry[active_leakage & (leakage_kind == 2)] = higher_hidden_index

        normal_indices = np.flatnonzero(~active_leakage)
        if normal_indices.size:
            ideal_pair = np.column_stack(
                (
                    np.full(normal_indices.size, g_ideal_index, dtype=np.int64),
                    np.where(error_now[normal_indices], e_ideal_index, g_ideal_index),
                )
            )
            for constituent in range(2):
                preparation_rows = observation_model.preparation_kernel[
                    carry[normal_indices], ideal_pair[:, constituent]
                ]
                hidden_pre = _sample_rows(preparation_rows, rng)
                observed = _sample_rows(observation_model.readout_confusion[hidden_pre], rng)
                reset_rows = observation_model.reset_kernel[observed, hidden_pre]
                carry[normal_indices] = _sample_rows(reset_rows, rng)
                if constituent == 0:
                    x_cycle[normal_indices] = observed
                else:
                    z_cycle[normal_indices] = observed

        if absolute_cycle >= actual.burn_in_cycles:
            cycle = absolute_cycle - actual.burn_in_cycles
            hidden_code[:, cycle] = code_now
            hidden_depth[:, cycle] = depth_at_cycle_start.astype(np.int16)
            hidden_leakage[:, cycle] = leakage_kind
            observed_x[:, cycle] = x_cycle.astype(np.int8)
            observed_z[:, cycle] = z_cycle.astype(np.int8)

        leakage_remaining[active_leakage] -= 1
        ended = leakage_remaining == 0
        leakage_kind[ended] = 0
        # higher spontaneous decay to addressed manifold followed by reset is external
        # to the T2.0.3 controller kernel; reset carry when the event ends.
        carry[ended & active_leakage] = g_hidden_index

    g_observed_index = OBSERVED_CLASSES.index("g")
    all_gg = (observed_x == g_observed_index) & (observed_z == g_observed_index)
    observed_leakage = (observed_x == leakage_observed_index) | (
        observed_z == leakage_observed_index
    )
    non_g_activity = ~all_gg
    arrays = (
        hidden_code,
        hidden_depth,
        hidden_leakage,
        observed_x,
        observed_z,
        all_gg,
        non_g_activity,
        observed_leakage,
    )
    for array in arrays:
        array.setflags(write=False)
    return OccupancyDataset(
        hidden_code_occupied=hidden_code,
        hidden_error_depth=hidden_depth,
        hidden_leakage_kind=hidden_leakage,
        observed_x=observed_x,
        observed_z=observed_z,
        all_gg=all_gg,
        non_g_activity=non_g_activity,
        observed_leakage=observed_leakage,
        seed=actual.seed,
    )


def _window_success_counts(
    all_gg: NDArray[np.bool_], lengths: Sequence[int]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    cumulative = np.pad(
        np.cumsum(all_gg, axis=1, dtype=np.int64),
        ((0, 0), (1, 0)),
        mode="constant",
    )
    successes = np.empty((all_gg.shape[0], len(lengths)), dtype=np.int64)
    totals = np.empty(len(lengths), dtype=np.int64)
    for index, length in enumerate(lengths):
        windows = cumulative[:, length:] - cumulative[:, :-length]
        successes[:, index] = np.sum(windows == length, axis=1)
        totals[index] = windows.shape[1]
    return successes, totals


def _fit_all_gg(
    lengths: NDArray[np.float64], probabilities: NDArray[np.float64]
) -> tuple[float, float, float, float, float]:
    if probabilities.shape != lengths.shape or np.any(probabilities <= 0.0):
        raise ValueError("all-gg probabilities must be positive and match lengths")
    slope, intercept = np.polyfit(lengths, np.log(probabilities), 1)
    fitted_lambda = float(np.exp(slope))
    fitted_a = float(np.exp(intercept))
    if not 0.0 < fitted_lambda < 1.0:
        raise ValueError("fitted lambda must lie strictly in (0, 1)")
    p_error = 1.0 - fitted_lambda
    occupancy = fitted_a * fitted_lambda
    prediction = intercept + slope * lengths
    residual = float(np.sum((np.log(probabilities) - prediction) ** 2))
    total = float(np.sum((np.log(probabilities) - np.mean(np.log(probabilities))) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return fitted_a, fitted_lambda, p_error, occupancy, r_squared


def estimate_occupancy_from_syndrome(
    observed_all_gg: ArrayLike,
    *,
    string_lengths: Sequence[int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> SyndromeOccupancyEstimate:
    """只用 observed all-gg matrix 估计 occupancy；接口无 hidden truth 参数。"""

    all_gg = np.asarray(observed_all_gg)
    if all_gg.ndim != 2 or all_gg.dtype != np.bool_:
        raise TypeError("observed_all_gg must be a 2D boolean matrix")
    if all_gg.shape[0] < 50 or all_gg.shape[1] < 20:
        raise ValueError("observed_all_gg dataset is too small for the registered fit")
    lengths_tuple = tuple(int(value) for value in string_lengths)
    if len(lengths_tuple) < 4 or any(
        value < 1 or value >= all_gg.shape[1] for value in lengths_tuple
    ):
        raise ValueError("string_lengths require at least four valid window lengths")
    if any(right <= left for left, right in zip(lengths_tuple, lengths_tuple[1:])):
        raise ValueError("string_lengths must be strictly increasing")
    replicates = _integer(bootstrap_replicates, "bootstrap_replicates", 100)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, (int, np.integer)):
        raise TypeError("bootstrap_seed must be an integer")
    confidence = _probability(confidence_level, "confidence_level", strict=True)

    successes, totals = _window_success_counts(all_gg, lengths_tuple)
    probabilities = np.sum(successes, axis=0) / (all_gg.shape[0] * totals)
    lengths = np.asarray(lengths_tuple, dtype=np.float64)
    fitted_a, fitted_lambda, p_error, occupancy, r_squared = _fit_all_gg(
        lengths, probabilities
    )

    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_occupancy = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, all_gg.shape[0], size=all_gg.shape[0])
        sampled = np.sum(successes[indices], axis=0) / (indices.size * totals)
        bootstrap_occupancy[replicate] = _fit_all_gg(lengths, sampled)[3]
    tail = 0.5 * (1.0 - confidence)
    statistical_low, statistical_high = np.quantile(
        bootstrap_occupancy, [tail, 1.0 - tail]
    )
    # S4E explicitly estimates first-order model corrections as p_err^2.
    model_bound = p_error**2
    combined = (
        max(0.0, float(statistical_low) - model_bound),
        min(1.0, float(statistical_high) + model_bound),
    )
    return SyndromeOccupancyEstimate(
        string_lengths=lengths_tuple,
        all_gg_probabilities=tuple(float(value) for value in probabilities),
        fitted_a=fitted_a,
        fitted_lambda=fitted_lambda,
        physical_error_probability=p_error,
        occupancy=occupancy,
        occupancy_statistical_ci=(float(statistical_low), float(statistical_high)),
        first_order_model_error_bound=model_bound,
        occupancy_combined_ci=combined,
        r_squared_log_probability=r_squared,
    )


def _bootstrap_hidden_occupancy(
    hidden_code: NDArray[np.bool_], config: OccupancyCorrelationConfig
) -> IntervalEstimate:
    per_shot = np.mean(hidden_code, axis=1)
    rng = np.random.default_rng(config.bootstrap_seed + 1)
    means = np.empty(config.bootstrap_replicates, dtype=np.float64)
    for replicate in range(config.bootstrap_replicates):
        indices = rng.integers(0, config.shots, size=config.shots)
        means[replicate] = np.mean(per_shot[indices])
    tail = 0.5 * (1.0 - config.confidence_level)
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return IntervalEstimate(
        mean=float(np.mean(per_shot)),
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=config.confidence_level,
        method="shot_cluster_bootstrap",
        replicates=config.bootstrap_replicates,
    )


def _max_run(values: NDArray[np.bool_]) -> NDArray[np.int64]:
    current = np.zeros(values.shape[0], dtype=np.int64)
    maximum = np.zeros(values.shape[0], dtype=np.int64)
    for cycle in range(values.shape[1]):
        current = np.where(values[:, cycle], current + 1, 0)
        maximum = np.maximum(maximum, current)
    return maximum


def _lag_sufficient_statistics(
    activity: NDArray[np.bool_], lags: Sequence[int]
) -> NDArray[np.float64]:
    statistics = np.empty((activity.shape[0], len(lags), 4), dtype=np.float64)
    for index, lag in enumerate(lags):
        x = activity[:, :-lag]
        y = activity[:, lag:]
        statistics[:, index, 0] = np.sum(x, axis=1)
        statistics[:, index, 1] = np.sum(y, axis=1)
        statistics[:, index, 2] = np.sum(x & y, axis=1)
        statistics[:, index, 3] = x.shape[1]
    return statistics


def _pooled_correlations(statistics: NDArray[np.float64]) -> NDArray[np.float64]:
    if statistics.ndim != 3 or statistics.shape[0] == 0:
        raise ValueError("statistics must contain at least one shot")
    sum_x = np.sum(statistics[:, :, 0], axis=0)
    sum_y = np.sum(statistics[:, :, 1], axis=0)
    sum_xy = np.sum(statistics[:, :, 2], axis=0)
    count = np.sum(statistics[:, :, 3], axis=0)
    mean_x = sum_x / count
    mean_y = sum_y / count
    covariance = sum_xy / count - mean_x * mean_y
    denominator = np.sqrt(mean_x * (1.0 - mean_x) * mean_y * (1.0 - mean_y))
    if np.any(denominator <= 0.0):
        raise ValueError("activity variance must be positive at every registered lag")
    return covariance / denominator


def estimate_leakage_tail_correlation(
    observed_activity: ArrayLike,
    observed_leakage: ArrayLike,
    *,
    lags: Sequence[int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> TailCorrelationEstimate:
    """计算 pooled lag correlation 与 leakage-run>=2 trajectory removal。"""

    activity = np.asarray(observed_activity)
    leakage = np.asarray(observed_leakage)
    if activity.dtype != np.bool_ or leakage.dtype != np.bool_:
        raise TypeError("observed_activity and observed_leakage must be boolean matrices")
    if activity.ndim != 2 or leakage.shape != activity.shape:
        raise ValueError("observed matrices must have the same 2D shape")
    lag_tuple = tuple(int(value) for value in lags)
    if len(lag_tuple) < 4 or any(value < 1 or value >= activity.shape[1] for value in lag_tuple):
        raise ValueError("lags require at least four valid positive values")
    replicates = _integer(bootstrap_replicates, "bootstrap_replicates", 100)
    confidence = _probability(confidence_level, "confidence_level", strict=True)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, (int, np.integer)):
        raise TypeError("bootstrap_seed must be an integer")

    keep = _max_run(leakage) < 2
    retained = int(np.count_nonzero(keep))
    if retained < 50:
        raise ValueError("leakage removal retained too few trajectories")
    statistics = _lag_sufficient_statistics(activity, lag_tuple)
    before = _pooled_correlations(statistics)
    after = _pooled_correlations(statistics[keep])
    mean_before = float(np.mean(before))
    mean_after = float(np.mean(after))

    rng = np.random.default_rng(int(bootstrap_seed))
    boot_before = np.empty(replicates, dtype=np.float64)
    boot_after = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, activity.shape[0], size=activity.shape[0])
        sampled_keep = keep[indices]
        if np.count_nonzero(sampled_keep) < 20:
            raise RuntimeError("bootstrap leakage removal retained too few trajectories")
        boot_before[replicate] = float(np.mean(_pooled_correlations(statistics[indices])))
        boot_after[replicate] = float(
            np.mean(_pooled_correlations(statistics[indices][sampled_keep]))
        )
    boot_difference = boot_before - boot_after
    tail = 0.5 * (1.0 - confidence)

    def interval(mean: float, values: NDArray[np.float64], method: str) -> IntervalEstimate:
        low, high = np.quantile(values, [tail, 1.0 - tail])
        return IntervalEstimate(
            mean=mean,
            ci_low=float(low),
            ci_high=float(high),
            confidence_level=confidence,
            method=method,
            replicates=replicates,
        )

    shrink_ratio = mean_before / max(abs(mean_after), 1.0e-12)
    return TailCorrelationEstimate(
        lags=lag_tuple,
        before_removal=tuple(float(value) for value in before),
        after_removal=tuple(float(value) for value in after),
        mean_before=interval(mean_before, boot_before, "shot_cluster_bootstrap"),
        mean_after=interval(mean_after, boot_after, "postselected_shot_cluster_bootstrap"),
        paired_difference=interval(
            mean_before - mean_after,
            boot_difference,
            "paired_shot_cluster_bootstrap",
        ),
        shrink_ratio=shrink_ratio,
        retained_shots=retained,
        retained_fraction=retained / activity.shape[0],
    )


def _evaluate_gate(
    config: OccupancyCorrelationConfig,
    hidden: IntervalEstimate,
    syndrome: SyndromeOccupancyEstimate,
    tail: TailCorrelationEstimate,
) -> OccupancyCorrelationGate:
    truth_difference = abs(hidden.mean - syndrome.occupancy)
    reference_difference = abs(hidden.mean - config.reference_occupancy)
    p_error_difference = abs(
        syndrome.physical_error_probability - config.physical_error_probability
    )
    combined_contains_truth = (
        syndrome.occupancy_combined_ci[0]
        <= hidden.mean
        <= syndrome.occupancy_combined_ci[1]
    )
    monotone_margin = float(
        np.min(-np.diff(np.asarray(syndrome.all_gg_probabilities, dtype=np.float64)))
    )
    checks = (
        OccupancyCorrelationCheck(
            "hidden_occupancy_near_reference",
            reference_difference <= config.maximum_reference_occupancy_error,
            reference_difference,
            "abs(hidden-reference) <= limit",
            config.maximum_reference_occupancy_error,
            f"reference={config.reference_occupancy}",
        ),
        OccupancyCorrelationCheck(
            "syndrome_estimate_matches_hidden_truth",
            truth_difference <= config.maximum_truth_syndrome_difference,
            truth_difference,
            "abs(syndrome-hidden) <= limit",
            config.maximum_truth_syndrome_difference,
            "same shared trajectories; estimator observed-only",
        ),
        OccupancyCorrelationCheck(
            "hidden_truth_inside_first_order_syndrome_interval",
            combined_contains_truth,
            hidden.mean,
            "combined syndrome CI contains hidden mean",
            syndrome.occupancy_combined_ci[1],
            f"combined_ci={syndrome.occupancy_combined_ci}",
        ),
        OccupancyCorrelationCheck(
            "physical_error_estimate_matches_generator",
            p_error_difference <= config.maximum_physical_error_estimate_error,
            p_error_difference,
            "abs(estimated-configured) <= limit",
            config.maximum_physical_error_estimate_error,
            f"estimated={syndrome.physical_error_probability}",
        ),
        OccupancyCorrelationCheck(
            "single_exponential_all_gg_fit",
            syndrome.r_squared_log_probability >= config.minimum_all_gg_fit_r_squared,
            syndrome.r_squared_log_probability,
            "log-probability R^2 >= limit",
            config.minimum_all_gg_fit_r_squared,
            "registered sliding-window lengths",
        ),
        OccupancyCorrelationCheck(
            "all_gg_probability_strictly_decreases",
            monotone_margin > 0.0,
            monotone_margin,
            "minimum adjacent decrease > 0",
            0.0,
            "rejects malformed/nondecaying strings",
        ),
        OccupancyCorrelationCheck(
            "leakage_removal_retains_expected_fraction",
            config.minimum_retained_fraction
            <= tail.retained_fraction
            <= config.maximum_retained_fraction,
            tail.retained_fraction,
            "minimum <= retained fraction <= maximum",
            config.minimum_retained_fraction,
            f"maximum={config.maximum_retained_fraction}",
        ),
        OccupancyCorrelationCheck(
            "long_lag_tail_detected_before_removal",
            tail.mean_before.mean >= config.minimum_pre_removal_tail_correlation,
            tail.mean_before.mean,
            "mean registered-lag correlation >= limit",
            config.minimum_pre_removal_tail_correlation,
            f"lags={tail.lags}",
        ),
        OccupancyCorrelationCheck(
            "tail_shrink_paired_ci_positive",
            tail.paired_difference.ci_low >= config.minimum_tail_difference_ci_low,
            tail.paired_difference.ci_low,
            "paired difference CI-low >= limit",
            config.minimum_tail_difference_ci_low,
            "before minus leakage-removed",
        ),
        OccupancyCorrelationCheck(
            "tail_shrink_ratio",
            tail.shrink_ratio >= config.minimum_tail_shrink_ratio,
            tail.shrink_ratio,
            "before/after mean correlation >= limit",
            config.minimum_tail_shrink_ratio,
            "point-estimate effect size",
        ),
        OccupancyCorrelationCheck(
            "post_removal_tail_small",
            abs(tail.mean_after.mean) <= config.maximum_post_removal_tail_correlation,
            abs(tail.mean_after.mean),
            "abs(post-removal mean tail) <= limit",
            config.maximum_post_removal_tail_correlation,
            "remaining short-memory/noise floor",
        ),
    )
    return OccupancyCorrelationGate(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def run_occupancy_correlation_validation(
    config: OccupancyCorrelationConfig | None = None,
) -> OccupancyCorrelationResult:
    """运行 shared trajectory、双 occupancy estimate 与 leakage-tail analysis。"""

    actual = OccupancyCorrelationConfig() if config is None else config
    if not isinstance(actual, OccupancyCorrelationConfig):
        raise TypeError("config must be an OccupancyCorrelationConfig or None")
    dataset = simulate_occupancy_correlation_dataset(actual)
    hidden = _bootstrap_hidden_occupancy(dataset.hidden_code_occupied, actual)
    syndrome = estimate_occupancy_from_syndrome(
        dataset.all_gg,
        string_lengths=actual.all_gg_string_lengths,
        bootstrap_replicates=actual.bootstrap_replicates,
        bootstrap_seed=actual.bootstrap_seed,
        confidence_level=actual.confidence_level,
    )
    tail = estimate_leakage_tail_correlation(
        dataset.non_g_activity,
        dataset.observed_leakage,
        lags=actual.tail_lags,
        bootstrap_replicates=actual.bootstrap_replicates,
        bootstrap_seed=actual.bootstrap_seed + 2,
        confidence_level=actual.confidence_level,
    )
    return OccupancyCorrelationResult(
        config=actual,
        hidden_occupancy=hidden,
        syndrome_occupancy=syndrome,
        tail_correlation=tail,
        gate=_evaluate_gate(actual, hidden, syndrome, tail),
    )


def write_occupancy_correlation_report(
    result: OccupancyCorrelationResult,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    if not isinstance(result, OccupancyCorrelationResult):
        raise TypeError("result must be an OccupancyCorrelationResult")
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
            fieldnames=("lag", "correlation_before", "correlation_after"),
        )
        writer.writeheader()
        for lag, before, after in zip(
            result.tail_correlation.lags,
            result.tail_correlation.before_removal,
            result.tail_correlation.after_removal,
        ):
            writer.writerow(
                {
                    "lag": lag,
                    "correlation_before": before,
                    "correlation_after": after,
                }
            )
