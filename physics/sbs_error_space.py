"""sBs 的 protocol-aligned coarse-grained error-space instrument。

本模块把 Sivak 2023 的方向性 flow diagram 实现为 ``C_i`` population-level
quantum instrument：每个理想 full-cycle outcome ``K_gg/K_ge/K_eg/K_ee`` 是一个
CP branch，branch 内可包含多个 effective Kraus components。这样可在多个输入
error subspaces 汇入同一低层 subspace 时保持完全正性和迹保持。

重要边界：这不是实验脉冲、Fock-space quadrature Kraus operator 或装置校准模型。
它主动去除不同 ``C_i`` 之间的相干，只保留每个 ``C_i`` 内的逻辑二维密度矩阵和
error-hierarchy population flow。readout confusion、``|f>`` leakage/reset 和 cycle
timing 分别属于 T2.0.3/T2.0.4。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


SBS_PROTOCOL_ID = "PROTO-SBS-MAIN"
SBS_OUTCOMES = ("K_gg", "K_ge", "K_eg", "K_ee")
MODEL_SCOPE = "protocol_aligned_coarse_grained_error_space_instrument"

_DEPTH_DECREMENT = {
    "K_gg": 0,
    "K_ge": 1,
    "K_eg": 1,
    "K_ee": 2,
}
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
# 文献 full cycle 的 deterministic frame effect 是 X_L Z_L=-iY_L；全局相位
# 不影响 density matrix，所以直接保留按 constituent 顺序的 X @ Z。
FULL_CYCLE_LOGICAL_FLIP = _PAULI_X @ _PAULI_Z


def _validate_bit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer bit")
    integer = int(value)
    if integer not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return integer


@dataclass(frozen=True)
class PauliFrame:
    """软件追踪的 logical Pauli frame，``x/z`` 均为 GF(2) bit。"""

    x: int = 0
    z: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _validate_bit(self.x, "x"))
        object.__setattr__(self, "z", _validate_bit(self.z, "z"))

    def after_full_sbs_cycle(self) -> "PauliFrame":
        """记录 X constituent 与 Z constituent 的 deterministic logical flips。"""

        return self.after_x_constituent().after_z_constituent()

    def after_x_constituent(self) -> "PauliFrame":
        """记录 ``R_Delta^X`` 的 deterministic ``X_L`` flip。"""

        return PauliFrame(x=self.x ^ 1, z=self.z)

    def after_z_constituent(self) -> "PauliFrame":
        """记录 ``R_Delta^Z`` 的 deterministic ``Z_L`` flip。"""

        return PauliFrame(x=self.x, z=self.z ^ 1)

    @property
    def unitary(self) -> NDArray[np.complex128]:
        """返回该 frame 的二维 unitary；全局相位不作物理区分。"""

        unitary = np.eye(2, dtype=np.complex128)
        if self.x:
            unitary = _PAULI_X @ unitary
        if self.z:
            unitary = _PAULI_Z @ unitary
        return unitary


@dataclass(frozen=True)
class SBSInstrumentResult:
    """一次 full sBs effective instrument 的精确分支结果。"""

    branch_states: Mapping[str, NDArray[np.complex128]]
    probabilities: Mapping[str, float]
    unconditional_state: NDArray[np.complex128]
    cycle_frame_corrected_state: NDArray[np.complex128]
    input_frame: PauliFrame
    output_frame: PauliFrame
    trace_before: float
    trace_after: float
    model_scope: str = MODEL_SCOPE


@dataclass(frozen=True)
class SBSPopulationResult:
    """对 ``C_i`` population 应用一次 instrument 后的分支与无条件结果。"""

    branch_populations: Mapping[str, NDArray[np.float64]]
    probabilities: Mapping[str, float]
    unconditional_population: NDArray[np.float64]


@dataclass(frozen=True)
class SBSTrajectory:
    """理想 instrument outcome 的可复现 sample trajectory。"""

    outcomes: tuple[str, ...]
    subspaces: tuple[str, ...]
    depths: tuple[int, ...]
    frames: tuple[PauliFrame, ...]
    seed: int | None
    protocol_id: str = SBS_PROTOCOL_ID
    observation_scope: str = "ideal_kraus_branch_not_noisy_readout"


class SBSErrorSpaceInstrument:
    """按 ``K_gg/K_ge/K_eg/K_ee`` 分组的 CPTP effective instrument。

    ``transition_probabilities[outcome][target, source]`` 给出从 source ``C_i``
    到 target ``C_j`` 且产生该 ideal branch label 的联合概率。所有 outcome 与
    target 对固定 source 的概率和必须为 1。

    每个非零 transition 构造一个
    ``sqrt(p)|target><source| tensor (X_L Z_L)`` Kraus component。使用多个
    component 是有意的：coarse graining 已丢弃 error-subspace coherence，不能用
    一个错误的 many-to-one amplitude matrix 冒充 trace-preserving channel。
    """

    protocol_id = SBS_PROTOCOL_ID
    model_scope = MODEL_SCOPE
    preserves_inter_subspace_coherence = False
    device_calibrated = False

    def __init__(
        self,
        *,
        subspaces: Sequence[str],
        depths: Sequence[int],
        transition_probabilities: Mapping[str, ArrayLike],
        normalization_tolerance: float = 1.0e-12,
    ) -> None:
        labels = tuple(subspaces)
        if not labels:
            raise ValueError("at least one error subspace is required")
        if labels[0] != "C0":
            raise ValueError("the first subspace must be C0")
        if len(labels) != len(set(labels)):
            raise ValueError("error subspace labels must be unique")
        if any(re.fullmatch(r"C(?:0|[1-9][0-9]*)", label) is None for label in labels):
            raise ValueError("error subspace labels must use canonical C0, C1, ... names")

        depth_array = np.asarray(depths)
        if depth_array.shape != (len(labels),):
            raise ValueError("depths must have one entry per error subspace")
        if not np.issubdtype(depth_array.dtype, np.integer):
            raise TypeError("depths must contain integers")
        depth_array = depth_array.astype(np.int64)
        if np.any(depth_array < 0) or depth_array[0] != 0:
            raise ValueError("depths must be non-negative and C0 must have depth 0")
        if np.any((depth_array == 0) & (np.arange(len(labels)) != 0)):
            raise ValueError("C0 must be the only depth-0 subspace")

        if (
            not np.isfinite(normalization_tolerance)
            or normalization_tolerance <= 0.0
            or normalization_tolerance > 1.0e-8
        ):
            raise ValueError("normalization_tolerance must be finite and in (0, 1e-8]")
        if set(transition_probabilities) != set(SBS_OUTCOMES):
            raise ValueError(f"transition_probabilities must have exactly {SBS_OUTCOMES}")

        transitions: dict[str, NDArray[np.float64]] = {}
        size = len(labels)
        for outcome in SBS_OUTCOMES:
            matrix = np.asarray(transition_probabilities[outcome], dtype=np.float64)
            if matrix.shape != (size, size):
                raise ValueError(f"{outcome} transition matrix must have shape {(size, size)}")
            if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
                raise ValueError(f"{outcome} transition probabilities must be finite and non-negative")
            matrix = np.array(matrix, dtype=np.float64, copy=True)
            matrix.setflags(write=False)
            transitions[outcome] = matrix

        self._validate_transition_semantics(labels, depth_array, transitions, normalization_tolerance)
        self.subspaces = labels
        self.depths = depth_array.copy()
        self.depths.setflags(write=False)
        self.transition_probabilities = MappingProxyType(transitions)
        self.normalization_tolerance = float(normalization_tolerance)
        self._components = self._build_kraus_components()

    @staticmethod
    def _validate_transition_semantics(
        labels: tuple[str, ...],
        depths: NDArray[np.int64],
        transitions: Mapping[str, NDArray[np.float64]],
        tolerance: float,
    ) -> None:
        size = len(labels)
        for outcome, decrement in _DEPTH_DECREMENT.items():
            matrix = transitions[outcome]
            target_indices, source_indices = np.nonzero(matrix > 0.0)
            for target, source in zip(target_indices.tolist(), source_indices.tolist()):
                if outcome == "K_gg" and target != source:
                    raise ValueError("K_gg must keep the same C_i subspace")
                actual = int(depths[source] - depths[target])
                if actual != decrement:
                    raise ValueError(
                        f"{outcome} must lower recovery depth by exactly {decrement}; "
                        f"got {labels[source]}->{labels[target]} with decrement {actual}"
                    )

        total_by_source = np.zeros(size, dtype=np.float64)
        for matrix in transitions.values():
            total_by_source += np.sum(matrix, axis=0)
        if not np.allclose(total_by_source, 1.0, rtol=0.0, atol=tolerance):
            raise ValueError(
                "outcome/target transition probabilities must sum to 1 for every source; "
                f"got {total_by_source.tolist()}"
            )
        if abs(transitions["K_gg"][0, 0] - 1.0) > tolerance:
            raise ValueError("C0 must emit K_gg with probability 1 in the no-error limit")
        for outcome in ("K_ge", "K_eg", "K_ee"):
            if np.sum(transitions[outcome][:, 0]) > tolerance:
                raise ValueError("C0 cannot emit a recovery branch in the no-error limit")

    @property
    def hilbert_dimension(self) -> int:
        return 2 * len(self.subspaces)

    @property
    def kraus_components(self) -> Mapping[str, tuple[NDArray[np.complex128], ...]]:
        """返回只读 grouped effective Kraus components。"""

        return self._components

    def _build_kraus_components(
        self,
    ) -> Mapping[str, tuple[NDArray[np.complex128], ...]]:
        grouped: dict[str, tuple[NDArray[np.complex128], ...]] = {}
        size = len(self.subspaces)
        for outcome in SBS_OUTCOMES:
            components: list[NDArray[np.complex128]] = []
            matrix = self.transition_probabilities[outcome]
            for target, source in zip(*np.nonzero(matrix > 0.0)):
                error_map = np.zeros((size, size), dtype=np.complex128)
                error_map[target, source] = np.sqrt(matrix[target, source])
                component = np.kron(error_map, FULL_CYCLE_LOGICAL_FLIP)
                component.setflags(write=False)
                components.append(component)
            grouped[outcome] = tuple(components)
        return MappingProxyType(grouped)

    def completeness_matrix(self) -> NDArray[np.complex128]:
        """返回 ``sum M^dagger M``，应等于 Hilbert-space identity。"""

        total = np.zeros((self.hilbert_dimension, self.hilbert_dimension), dtype=np.complex128)
        for components in self.kraus_components.values():
            for component in components:
                total += component.conj().T @ component
        return total

    def completeness_residual_norm(self) -> float:
        identity = np.eye(self.hilbert_dimension, dtype=np.complex128)
        return float(np.linalg.norm(self.completeness_matrix() - identity, ord=2))

    def _validate_density_matrix(self, state: ArrayLike) -> NDArray[np.complex128]:
        rho = np.asarray(state, dtype=np.complex128)
        expected = (self.hilbert_dimension, self.hilbert_dimension)
        if rho.shape != expected:
            raise ValueError(f"state must have shape {expected}")
        if not np.all(np.isfinite(rho)):
            raise ValueError("state must contain finite entries")
        if not np.allclose(rho, rho.conj().T, rtol=0.0, atol=1.0e-12):
            raise ValueError("state must be Hermitian")
        trace = complex(np.trace(rho))
        if abs(trace.imag) > 1.0e-12 or abs(trace.real - 1.0) > 1.0e-12:
            raise ValueError("state must have unit real trace")
        eigenvalues = np.linalg.eigvalsh(rho)
        if float(np.min(eigenvalues)) < -1.0e-11:
            raise ValueError("state must be positive semidefinite")
        return rho

    def apply_density_matrix(
        self,
        state: ArrayLike,
        *,
        frame: PauliFrame | None = None,
    ) -> SBSInstrumentResult:
        """应用 full-cycle effective instrument 并返回所有 ideal branch。

        ``branch_states`` 是 subnormalized density matrices；其 trace 即 outcome
        probability。``cycle_frame_corrected_state`` 只撤销本 cycle 的 deterministic
        X/Z logical flip，error-space population flow 保持不变。
        """

        rho = self._validate_density_matrix(state)
        input_frame = PauliFrame() if frame is None else frame
        if not isinstance(input_frame, PauliFrame):
            raise TypeError("frame must be a PauliFrame")

        branch_states: dict[str, NDArray[np.complex128]] = {}
        probabilities: dict[str, float] = {}
        unconditional = np.zeros_like(rho)
        for outcome in SBS_OUTCOMES:
            branch = np.zeros_like(rho)
            for component in self.kraus_components[outcome]:
                branch += component @ rho @ component.conj().T
            probability = float(np.trace(branch).real)
            if probability < -1.0e-12:
                raise RuntimeError("a CP branch produced a negative probability")
            branch_states[outcome] = branch
            probabilities[outcome] = max(0.0, probability)
            unconditional += branch

        trace_after = float(np.trace(unconditional).real)
        if abs(trace_after - 1.0) > 10.0 * self.normalization_tolerance:
            raise RuntimeError(f"instrument is not trace preserving: trace={trace_after}")

        cycle_unitary = np.kron(
            np.eye(len(self.subspaces), dtype=np.complex128),
            FULL_CYCLE_LOGICAL_FLIP,
        )
        corrected = cycle_unitary.conj().T @ unconditional @ cycle_unitary
        return SBSInstrumentResult(
            branch_states=MappingProxyType(branch_states),
            probabilities=MappingProxyType(probabilities),
            unconditional_state=unconditional,
            cycle_frame_corrected_state=corrected,
            input_frame=input_frame,
            output_frame=input_frame.after_full_sbs_cycle(),
            trace_before=float(np.trace(rho).real),
            trace_after=trace_after,
        )

    def _validate_population(self, population: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(population, dtype=np.float64)
        if values.shape != (len(self.subspaces),):
            raise ValueError("population must have one entry per error subspace")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("population must be finite and non-negative")
        total = float(np.sum(values))
        if abs(total - 1.0) > 1.0e-12:
            raise ValueError("population must sum to 1")
        return values

    def apply_population(self, population: ArrayLike) -> SBSPopulationResult:
        """应用同一 instrument 的 classical ``C_i`` population marginal。"""

        values = self._validate_population(population)
        branches: dict[str, NDArray[np.float64]] = {}
        probabilities: dict[str, float] = {}
        unconditional = np.zeros_like(values)
        for outcome in SBS_OUTCOMES:
            branch = self.transition_probabilities[outcome] @ values
            branches[outcome] = branch
            probabilities[outcome] = float(np.sum(branch))
            unconditional += branch
        if abs(float(np.sum(unconditional)) - 1.0) > 10.0 * self.normalization_tolerance:
            raise RuntimeError("population transition is not normalized")
        return SBSPopulationResult(
            branch_populations=MappingProxyType(branches),
            probabilities=MappingProxyType(probabilities),
            unconditional_population=unconditional,
        )

    def expected_depth(self, population: ArrayLike) -> float:
        values = self._validate_population(population)
        return float(np.dot(self.depths.astype(np.float64), values))

    def sample_trajectory(
        self,
        *,
        initial_subspace: str,
        cycles: int,
        seed: int | None = None,
        initial_frame: PauliFrame | None = None,
    ) -> SBSTrajectory:
        """sample ideal Kraus-branch trajectory；不含 noisy readout/reset。"""

        if initial_subspace not in self.subspaces:
            raise ValueError(f"unknown initial_subspace {initial_subspace!r}")
        if isinstance(cycles, bool) or not isinstance(cycles, (int, np.integer)):
            raise TypeError("cycles must be an integer")
        cycles = int(cycles)
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
            raise TypeError("seed must be an integer or None")
        normalized_seed = None if seed is None else int(seed)
        rng = np.random.default_rng(normalized_seed)
        frame = PauliFrame() if initial_frame is None else initial_frame
        if not isinstance(frame, PauliFrame):
            raise TypeError("initial_frame must be a PauliFrame")

        source = self.subspaces.index(initial_subspace)
        outcomes: list[str] = []
        subspaces = [self.subspaces[source]]
        depths = [int(self.depths[source])]
        frames = [frame]
        size = len(self.subspaces)
        for _ in range(cycles):
            probabilities: list[float] = []
            choices: list[tuple[str, int]] = []
            for outcome in SBS_OUTCOMES:
                for target in range(size):
                    probability = float(self.transition_probabilities[outcome][target, source])
                    if probability > 0.0:
                        choices.append((outcome, target))
                        probabilities.append(probability)
            probability_array = np.asarray(probabilities, dtype=np.float64)
            probability_array /= np.sum(probability_array)
            choice_index = int(rng.choice(len(choices), p=probability_array))
            outcome, source = choices[choice_index]
            frame = frame.after_full_sbs_cycle()
            outcomes.append(outcome)
            subspaces.append(self.subspaces[source])
            depths.append(int(self.depths[source]))
            frames.append(frame)

        return SBSTrajectory(
            outcomes=tuple(outcomes),
            subspaces=tuple(subspaces),
            depths=tuple(depths),
            frames=tuple(frames),
            seed=normalized_seed,
        )


def make_trickle_down_chain(
    *,
    max_depth: int,
    one_step_probability: float,
    two_step_probability: float,
    ge_fraction: float = 0.5,
) -> SBSErrorSpaceInstrument:
    """构造显式建模假设下的 ``C0...Cmax`` recovery-depth chain。

    对 ``C_d``：``K_gg`` 保持 d，``K_ge/K_eg`` 以给定比例降到 d-1，
    ``K_ee`` 在 d>=2 时降到 d-2。C0 强制 ``K_gg=1``；C1 不存在 two-step
    branch，未使用的 ``two_step_probability`` 不会被重分配或裁剪。

    三个概率参数是项目 modeling assumptions，不是 Sivak 装置校准值。
    """

    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)):
        raise TypeError("max_depth must be an integer")
    max_depth = int(max_depth)
    if max_depth < 2:
        raise ValueError("max_depth must be at least 2 to represent K_ee trickle-down")
    parameters = {
        "one_step_probability": one_step_probability,
        "two_step_probability": two_step_probability,
        "ge_fraction": ge_fraction,
    }
    validated: dict[str, float] = {}
    for name, value in parameters.items():
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real probability")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real probability") from exc
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        validated[name] = number
    p_one = validated["one_step_probability"]
    p_two = validated["two_step_probability"]
    fraction = validated["ge_fraction"]
    if p_one + p_two > 1.0 + 1.0e-15:
        raise ValueError("one_step_probability + two_step_probability must not exceed 1")

    size = max_depth + 1
    transitions = {
        outcome: np.zeros((size, size), dtype=np.float64) for outcome in SBS_OUTCOMES
    }
    for depth in range(size):
        one = p_one if depth >= 1 else 0.0
        two = p_two if depth >= 2 else 0.0
        transitions["K_gg"][depth, depth] = 1.0 - one - two
        if depth >= 1:
            transitions["K_ge"][depth - 1, depth] = one * fraction
            transitions["K_eg"][depth - 1, depth] = one * (1.0 - fraction)
        if depth >= 2:
            transitions["K_ee"][depth - 2, depth] = two

    return SBSErrorSpaceInstrument(
        subspaces=tuple(f"C{depth}" for depth in range(size)),
        depths=tuple(range(size)),
        transition_probabilities=transitions,
    )


def embed_logical_state(
    logical_state: ArrayLike,
    *,
    subspace_index: int,
    subspace_count: int,
) -> NDArray[np.complex128]:
    """把二维 logical density matrix 放入一个 ``C_i`` block。"""

    if isinstance(subspace_count, bool) or not isinstance(subspace_count, (int, np.integer)):
        raise TypeError("subspace_count must be an integer")
    if isinstance(subspace_index, bool) or not isinstance(subspace_index, (int, np.integer)):
        raise TypeError("subspace_index must be an integer")
    subspace_count = int(subspace_count)
    subspace_index = int(subspace_index)
    if subspace_count <= 0 or not 0 <= subspace_index < subspace_count:
        raise ValueError("subspace_index must lie within a positive subspace_count")
    logical = np.asarray(logical_state, dtype=np.complex128)
    if logical.shape != (2, 2):
        raise ValueError("logical_state must have shape (2, 2)")
    if not np.all(np.isfinite(logical)) or not np.allclose(
        logical, logical.conj().T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("logical_state must be finite and Hermitian")
    if abs(complex(np.trace(logical)) - 1.0) > 1.0e-12:
        raise ValueError("logical_state must have unit trace")
    if float(np.min(np.linalg.eigvalsh(logical))) < -1.0e-11:
        raise ValueError("logical_state must be positive semidefinite")
    state = np.zeros((2 * subspace_count, 2 * subspace_count), dtype=np.complex128)
    block = slice(2 * subspace_index, 2 * subspace_index + 2)
    state[block, block] = logical
    return state
