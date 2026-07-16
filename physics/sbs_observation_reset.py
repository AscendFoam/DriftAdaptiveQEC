"""sBs ideal Kraus branch 之上的 g/e/leakage observation 与 reset 模型。

模块严格分离四层：

1. T2.0.2 给出的 ideal full-cycle Kraus label；
2. ancilla hidden physical state ``g/e/f/higher``；
3. deployable observed class ``g/e/leakage``；
4. 基于 observed class 选择的 conditional reset 与 reset 后 hidden carry state。

所有 confusion/preparation/reset kernels 必须是显式、带 provenance 的 row-stochastic
输入；本模块不从 Sivak 文献中只有 F_g/F_e 对角元的 partial matrix 猜测 F_f 或
off-diagonal values。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sbs_error_space import SBS_OUTCOMES, SBS_PROTOCOL_ID


IDEAL_ANCILLA_STATES = ("g", "e")
HIDDEN_ANCILLA_STATES = ("g", "e", "f", "higher")
OBSERVED_CLASSES = ("g", "e", "leakage")
CONSTITUENT_ORDER = ("X", "Z")
OBSERVATION_MODEL_SCOPE = "protocol_aligned_hidden_observed_reset_model_not_device_calibrated"


def _require_nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _validate_counter(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _validate_state(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return value


def _validate_row_stochastic(
    values: ArrayLike,
    *,
    shape: tuple[int, ...],
    row_axis: int,
    name: str,
    tolerance: float = 1.0e-12,
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    row_sums = np.sum(matrix, axis=row_axis)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} rows must sum to 1; got {row_sums.tolist()}")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PairedSyndrome:
    """按真实执行顺序保存 X/Z constituent outcome 或 observed class。"""

    x: str
    z: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _validate_state(self.x, OBSERVED_CLASSES, "x"))
        object.__setattr__(self, "z", _validate_state(self.z, OBSERVED_CLASSES, "z"))

    def as_tuple(self) -> tuple[str, str]:
        return self.x, self.z


@dataclass(frozen=True)
class HiddenAncillaMemory:
    """只供 simulator truth lane 使用的 hidden ancilla memory。"""

    carry_state: str = "g"
    higher_run: int = 0
    f_run: int = 0
    constituent_index: int = 0
    cycle_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "carry_state",
            _validate_state(self.carry_state, HIDDEN_ANCILLA_STATES, "carry_state"),
        )
        for name in ("higher_run", "f_run", "constituent_index", "cycle_index"):
            object.__setattr__(self, name, _validate_counter(getattr(self, name), name))


@dataclass(frozen=True)
class ObservedSyndromeMemory:
    """可部署 lane 允许消费的 observed counters；不含 hidden truth。"""

    x_e_run: int = 0
    z_e_run: int = 0
    leakage_constituent_run: int = 0
    leakage_cycle_run: int = 0
    cycle_index: int = 0
    last_syndrome: PairedSyndrome | None = None

    def __post_init__(self) -> None:
        for name in (
            "x_e_run",
            "z_e_run",
            "leakage_constituent_run",
            "leakage_cycle_run",
            "cycle_index",
        ):
            object.__setattr__(self, name, _validate_counter(getattr(self, name), name))
        if self.last_syndrome is not None:
            if not isinstance(self.last_syndrome, PairedSyndrome):
                raise TypeError("last_syndrome must be a PairedSyndrome or None")
            _validate_state(self.last_syndrome.x, OBSERVED_CLASSES, "last_syndrome.x")
            _validate_state(self.last_syndrome.z, OBSERVED_CLASSES, "last_syndrome.z")


@dataclass(frozen=True)
class SBSObservedCycle:
    """可部署消费者视图；结构上不暴露 ideal/hidden states。"""

    cycle_index: int
    syndrome: PairedSyndrome
    reset_actions: tuple[str, str]
    x_e_run: int
    z_e_run: int
    leakage_constituent_run: int
    leakage_cycle_run: int
    observation_scope: str = "noisy_classified_observation"

    def as_deployable_dict(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "syndrome_x": self.syndrome.x,
            "syndrome_z": self.syndrome.z,
            "reset_action_x": self.reset_actions[0],
            "reset_action_z": self.reset_actions[1],
            "x_e_run": self.x_e_run,
            "z_e_run": self.z_e_run,
            "leakage_constituent_run": self.leakage_constituent_run,
            "leakage_cycle_run": self.leakage_cycle_run,
            "observation_scope": self.observation_scope,
        }


@dataclass(frozen=True)
class SBSTruthCycle:
    """仅诊断/验证使用的 truth view。"""

    cycle_index: int
    ideal_kraus_label: str
    ideal_syndrome: PairedSyndrome
    hidden_pre_readout: tuple[str, str]
    hidden_post_reset: tuple[str, str]
    hidden_higher_run: int
    hidden_f_run: int
    hidden_leakage_event: bool
    truth_scope: str = "simulator_hidden_truth_not_deployable_input"


@dataclass(frozen=True)
class SBSObservationResetStep:
    observed: SBSObservedCycle
    truth: SBSTruthCycle
    hidden_memory: HiddenAncillaMemory
    observed_memory: ObservedSyndromeMemory


@dataclass(frozen=True)
class SBSObservationResetTrajectory:
    observed_cycles: tuple[SBSObservedCycle, ...]
    truth_cycles: tuple[SBSTruthCycle, ...]
    final_hidden_memory: HiddenAncillaMemory
    final_observed_memory: ObservedSyndromeMemory
    seed: int | None
    protocol_id: str = SBS_PROTOCOL_ID

    def deployable_records(self) -> tuple[dict[str, object], ...]:
        return tuple(cycle.as_deployable_dict() for cycle in self.observed_cycles)


def ideal_syndrome_from_kraus(label: str) -> PairedSyndrome:
    """把论文 Kraus label 转成真实执行顺序 ``(X,Z)``。

    文献定义 ``K_ge = K_g^Z K_e^X``，所以 label 字符顺序是 ``(Z,X)``；
    通道执行顺序为右侧 X constituent 先、左侧 Z constituent 后。
    """

    if label not in SBS_OUTCOMES:
        raise ValueError(f"ideal Kraus label must be one of {SBS_OUTCOMES}")
    z_outcome, x_outcome = label.removeprefix("K_")
    return PairedSyndrome(x=x_outcome, z=z_outcome)


class SBSObservationResetModel:
    """hidden/preparation -> observed -> conditional reset 的两-constituent模型。"""

    protocol_id = SBS_PROTOCOL_ID
    model_scope = OBSERVATION_MODEL_SCOPE
    device_calibrated = False
    exposes_hidden_truth_to_deployable_view = False

    def __init__(
        self,
        *,
        preparation_kernel: ArrayLike,
        readout_confusion: ArrayLike,
        reset_kernel: ArrayLike,
        reset_action_by_observation: Mapping[str, str],
        counter_max: int,
        preparation_provenance: str,
        readout_provenance: str,
        reset_provenance: str,
    ) -> None:
        # [carry hidden, ideal g/e, pre-readout hidden]
        self.preparation_kernel = _validate_row_stochastic(
            preparation_kernel,
            shape=(4, 2, 4),
            row_axis=2,
            name="preparation_kernel",
        )
        # [pre-readout hidden, observed g/e/leakage]
        self.readout_confusion = _validate_row_stochastic(
            readout_confusion,
            shape=(4, 3),
            row_axis=1,
            name="readout_confusion",
        )
        # [observed class, pre-readout hidden, post-reset hidden]
        self.reset_kernel = _validate_row_stochastic(
            reset_kernel,
            shape=(3, 4, 4),
            row_axis=2,
            name="reset_kernel",
        )
        if set(reset_action_by_observation) != set(OBSERVED_CLASSES):
            raise ValueError(
                f"reset_action_by_observation must have exactly {OBSERVED_CLASSES}"
            )
        actions = {
            observed: _require_nonempty_text(reset_action_by_observation[observed], f"action[{observed}]")
            for observed in OBSERVED_CLASSES
        }
        self.reset_action_by_observation = MappingProxyType(actions)
        self.counter_max = _validate_counter(counter_max, "counter_max")
        if self.counter_max < 1:
            raise ValueError("counter_max must be at least 1")
        self.preparation_provenance = _require_nonempty_text(
            preparation_provenance, "preparation_provenance"
        )
        self.readout_provenance = _require_nonempty_text(
            readout_provenance, "readout_provenance"
        )
        self.reset_provenance = _require_nonempty_text(reset_provenance, "reset_provenance")

    def _saturating_increment(self, value: int) -> int:
        return min(self.counter_max, value + 1)

    @staticmethod
    def _sample(probabilities: NDArray[np.float64], rng: np.random.Generator) -> int:
        return int(rng.choice(probabilities.size, p=probabilities))

    def _step_with_rng(
        self,
        ideal_kraus_label: str,
        *,
        hidden_memory: HiddenAncillaMemory,
        observed_memory: ObservedSyndromeMemory,
        rng: np.random.Generator,
    ) -> SBSObservationResetStep:
        if not isinstance(hidden_memory, HiddenAncillaMemory):
            raise TypeError("hidden_memory must be HiddenAncillaMemory")
        if not isinstance(observed_memory, ObservedSyndromeMemory):
            raise TypeError("observed_memory must be ObservedSyndromeMemory")
        if hidden_memory.cycle_index != observed_memory.cycle_index:
            raise ValueError("hidden and observed memories must have the same cycle_index")
        ideal = ideal_syndrome_from_kraus(ideal_kraus_label)
        ideal_by_quadrature = {"X": ideal.x, "Z": ideal.z}

        carry = HIDDEN_ANCILLA_STATES.index(hidden_memory.carry_state)
        higher_run = hidden_memory.higher_run
        f_run = hidden_memory.f_run
        leakage_constituent_run = observed_memory.leakage_constituent_run
        hidden_pre: list[str] = []
        hidden_post: list[str] = []
        observed_values: list[str] = []
        reset_actions: list[str] = []

        for quadrature in CONSTITUENT_ORDER:
            ideal_index = IDEAL_ANCILLA_STATES.index(ideal_by_quadrature[quadrature])
            pre = self._sample(self.preparation_kernel[carry, ideal_index], rng)
            observed_index = self._sample(self.readout_confusion[pre], rng)
            post = self._sample(self.reset_kernel[observed_index, pre], rng)
            pre_name = HIDDEN_ANCILLA_STATES[pre]
            observed_name = OBSERVED_CLASSES[observed_index]
            post_name = HIDDEN_ANCILLA_STATES[post]

            higher_run = self._saturating_increment(higher_run) if pre_name == "higher" else 0
            f_run = self._saturating_increment(f_run) if pre_name == "f" else 0
            leakage_constituent_run = (
                self._saturating_increment(leakage_constituent_run)
                if observed_name == "leakage"
                else 0
            )
            hidden_pre.append(pre_name)
            hidden_post.append(post_name)
            observed_values.append(observed_name)
            reset_actions.append(self.reset_action_by_observation[observed_name])
            carry = post

        syndrome = PairedSyndrome(x=observed_values[0], z=observed_values[1])
        x_e_run = (
            self._saturating_increment(observed_memory.x_e_run) if syndrome.x == "e" else 0
        )
        z_e_run = (
            self._saturating_increment(observed_memory.z_e_run) if syndrome.z == "e" else 0
        )
        any_observed_leakage = "leakage" in syndrome.as_tuple()
        leakage_cycle_run = (
            self._saturating_increment(observed_memory.leakage_cycle_run)
            if any_observed_leakage
            else 0
        )
        cycle_index = hidden_memory.cycle_index
        next_hidden = HiddenAncillaMemory(
            carry_state=HIDDEN_ANCILLA_STATES[carry],
            higher_run=higher_run,
            f_run=f_run,
            constituent_index=hidden_memory.constituent_index + 2,
            cycle_index=cycle_index + 1,
        )
        next_observed = ObservedSyndromeMemory(
            x_e_run=x_e_run,
            z_e_run=z_e_run,
            leakage_constituent_run=leakage_constituent_run,
            leakage_cycle_run=leakage_cycle_run,
            cycle_index=cycle_index + 1,
            last_syndrome=syndrome,
        )
        observed_cycle = SBSObservedCycle(
            cycle_index=cycle_index,
            syndrome=syndrome,
            reset_actions=(reset_actions[0], reset_actions[1]),
            x_e_run=x_e_run,
            z_e_run=z_e_run,
            leakage_constituent_run=leakage_constituent_run,
            leakage_cycle_run=leakage_cycle_run,
        )
        truth_cycle = SBSTruthCycle(
            cycle_index=cycle_index,
            ideal_kraus_label=ideal_kraus_label,
            ideal_syndrome=ideal,
            hidden_pre_readout=(hidden_pre[0], hidden_pre[1]),
            hidden_post_reset=(hidden_post[0], hidden_post[1]),
            hidden_higher_run=higher_run,
            hidden_f_run=f_run,
            hidden_leakage_event=any(
                state in {"f", "higher"} for state in hidden_pre
            ),
        )
        return SBSObservationResetStep(
            observed=observed_cycle,
            truth=truth_cycle,
            hidden_memory=next_hidden,
            observed_memory=next_observed,
        )

    def step(
        self,
        ideal_kraus_label: str,
        *,
        hidden_memory: HiddenAncillaMemory | None = None,
        observed_memory: ObservedSyndromeMemory | None = None,
        seed: int | None = None,
    ) -> SBSObservationResetStep:
        """执行一个 full cycle；便捷接口只为单步复现使用。"""

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
            raise TypeError("seed must be an integer or None")
        rng = np.random.default_rng(None if seed is None else int(seed))
        return self._step_with_rng(
            ideal_kraus_label,
            hidden_memory=HiddenAncillaMemory() if hidden_memory is None else hidden_memory,
            observed_memory=(
                ObservedSyndromeMemory() if observed_memory is None else observed_memory
            ),
            rng=rng,
        )

    def simulate(
        self,
        ideal_kraus_labels: Sequence[str],
        *,
        seed: int | None,
        initial_hidden_memory: HiddenAncillaMemory | None = None,
        initial_observed_memory: ObservedSyndromeMemory | None = None,
    ) -> SBSObservationResetTrajectory:
        """用同一个 RNG 连续模拟 ideal branch sequence。"""

        if isinstance(ideal_kraus_labels, (str, bytes)):
            raise TypeError("ideal_kraus_labels must be a sequence of labels, not text")
        labels = tuple(ideal_kraus_labels)
        if any(label not in SBS_OUTCOMES for label in labels):
            raise ValueError(f"all ideal Kraus labels must be in {SBS_OUTCOMES}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
            raise TypeError("seed must be an integer or None")
        normalized_seed = None if seed is None else int(seed)
        rng = np.random.default_rng(normalized_seed)
        hidden = HiddenAncillaMemory() if initial_hidden_memory is None else initial_hidden_memory
        observed = (
            ObservedSyndromeMemory()
            if initial_observed_memory is None
            else initial_observed_memory
        )
        observed_cycles: list[SBSObservedCycle] = []
        truth_cycles: list[SBSTruthCycle] = []
        for label in labels:
            result = self._step_with_rng(
                label,
                hidden_memory=hidden,
                observed_memory=observed,
                rng=rng,
            )
            observed_cycles.append(result.observed)
            truth_cycles.append(result.truth)
            hidden = result.hidden_memory
            observed = result.observed_memory
        return SBSObservationResetTrajectory(
            observed_cycles=tuple(observed_cycles),
            truth_cycles=tuple(truth_cycles),
            final_hidden_memory=hidden,
            final_observed_memory=observed,
            seed=normalized_seed,
        )


def make_persistent_leakage_model(
    *,
    readout_confusion: ArrayLike,
    f_injection_given_g: float,
    f_injection_given_e: float,
    higher_injection_given_g: float,
    higher_injection_given_e: float,
    e_reset_success: float,
    f_reset_success: float,
    higher_reset_success: float,
    counter_max: int,
    readout_provenance: str,
    parameter_provenance: str,
) -> SBSObservationResetModel:
    """构造显式 assumption-driven persistent leakage/reset model。

    previous reset carry 为 g 时，ideal g/e 可注入 f/higher；carry 为 e/f/higher
    时先保持该 hidden state，模拟 reset 未清除的持久故障。观测 g 不发 reset，
    观测 e 只尝试 e->g，观测 leakage 尝试 f/higher->g。所有概率均须显式给出。
    """

    names_and_values = {
        "f_injection_given_g": f_injection_given_g,
        "f_injection_given_e": f_injection_given_e,
        "higher_injection_given_g": higher_injection_given_g,
        "higher_injection_given_e": higher_injection_given_e,
        "e_reset_success": e_reset_success,
        "f_reset_success": f_reset_success,
        "higher_reset_success": higher_reset_success,
    }
    parameters: dict[str, float] = {}
    for name, value in names_and_values.items():
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real probability")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real probability") from exc
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        parameters[name] = number
    for ideal in IDEAL_ANCILLA_STATES:
        suffix = "g" if ideal == "g" else "e"
        if (
            parameters[f"f_injection_given_{suffix}"]
            + parameters[f"higher_injection_given_{suffix}"]
            > 1.0 + 1.0e-15
        ):
            raise ValueError(f"f+higher injection probabilities given {ideal} must not exceed 1")

    preparation = np.zeros((4, 2, 4), dtype=np.float64)
    for ideal_index, ideal in enumerate(IDEAL_ANCILLA_STATES):
        suffix = "g" if ideal == "g" else "e"
        p_f = parameters[f"f_injection_given_{suffix}"]
        p_higher = parameters[f"higher_injection_given_{suffix}"]
        base_index = HIDDEN_ANCILLA_STATES.index(ideal)
        preparation[0, ideal_index, base_index] = 1.0 - p_f - p_higher
        preparation[0, ideal_index, 2] = p_f
        preparation[0, ideal_index, 3] = p_higher
    for carry in range(1, 4):
        preparation[carry, :, carry] = 1.0

    reset = np.zeros((3, 4, 4), dtype=np.float64)
    # 默认：任何不匹配 observed action 的 hidden state 保持不变。
    for observed in range(3):
        for hidden in range(4):
            reset[observed, hidden, hidden] = 1.0
    e_success = parameters["e_reset_success"]
    reset[1, 1, :] = 0.0
    reset[1, 1, 0] = e_success
    reset[1, 1, 1] = 1.0 - e_success
    f_success = parameters["f_reset_success"]
    reset[2, 2, :] = 0.0
    reset[2, 2, 0] = f_success
    reset[2, 2, 2] = 1.0 - f_success
    higher_success = parameters["higher_reset_success"]
    reset[2, 3, :] = 0.0
    reset[2, 3, 0] = higher_success
    reset[2, 3, 3] = 1.0 - higher_success

    parameter_source = _require_nonempty_text(parameter_provenance, "parameter_provenance")
    return SBSObservationResetModel(
        preparation_kernel=preparation,
        readout_confusion=readout_confusion,
        reset_kernel=reset,
        reset_action_by_observation={
            "g": "no_reset_pulse",
            "e": "conditional_e_to_g_reset",
            "leakage": "conditional_f_or_higher_reset_attempt",
        },
        counter_max=counter_max,
        preparation_provenance=parameter_source,
        readout_provenance=readout_provenance,
        reset_provenance=parameter_source,
    )
