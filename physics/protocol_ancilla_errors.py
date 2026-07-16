"""sBs 与 sharpen--trim 的协议原生 ancilla/readout/reset error layer。

本模块刻意不共享两个协议的观测字母表：

* sBs 复用 :mod:`physics.sbs_observation_reset` 的 ``g/e/leakage`` hidden-
  observed-reset kernel，并叠加 stage-resolved bit/phase flips；
* sharpen--trim 使用原生 ``+y/-y`` readout、四轮 action、conditional ``pi/2``
  reset 和独立 hidden carry。

Steane/Knill/qunaught/P-Steane 只保留不可执行的 theory/noise-shaping registry。
任何 fault probability、confusion matrix、reset success 或 backaction scale 都必须带
显式 provenance；这里没有 device-calibrated 默认值。
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import LATTICE_CONST
from .sbs_error_space import SBS_OUTCOMES, SBS_PROTOCOL_ID
from .sbs_observation_reset import (
    HIDDEN_ANCILLA_STATES,
    OBSERVED_CLASSES,
    HiddenAncillaMemory,
    ObservedSyndromeMemory,
    SBSObservationResetModel,
    SBSObservationResetStep,
    make_persistent_leakage_model,
)


SBS_FAULT_STAGES = ("small_cd", "big_cd", "readout")
SBS_CONSTITUENTS = ("X", "Z")
SHARPEN_TRIM_PROTOCOL_ID = "PROTO-SHARPEN-TRIM-XVAL"
SHARPEN_TRIM_ROUND_TYPES = (
    "q_peak_sharpen",
    "p_peak_sharpen",
    "q_envelope_trim",
    "p_envelope_trim",
)
SHARPEN_TRIM_HIDDEN_STATES = ("+y", "-y", "leakage")
SHARPEN_TRIM_OBSERVED_CLASSES = ("+y", "-y")
SHARPEN_TRIM_CARRY_STATES = ("g", "+y", "-y", "leakage")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _counter(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real probability")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real probability") from exc
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _probability_matrix(
    values: object,
    shape: tuple[int, int],
    name: str,
) -> tuple[tuple[float, ...], ...]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    return tuple(tuple(float(value) for value in row) for row in array)


def _probability_vector(values: object, length: int, name: str) -> tuple[float, ...]:
    try:
        vector = tuple(_probability(value, name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain {length} probabilities") from exc
    if len(vector) != length:
        raise ValueError(f"{name} must contain {length} probabilities")
    return vector


def _row_stochastic(values: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(np.sum(matrix, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} rows must sum to 1")
    result = np.array(matrix, copy=True)
    result.setflags(write=False)
    return result


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError("seed must be non-negative")
    return integer


def _toggle_ge(value: str) -> str:
    if value == "g":
        return "e"
    if value == "e":
        return "g"
    raise ValueError("sBs ideal value must be g or e")


def _toggle_y(value: str) -> str:
    if value == "+y":
        return "-y"
    if value == "-y":
        return "+y"
    raise ValueError("Y-basis value must be +y or -y")


def _kraus_to_xz(label: str) -> list[str]:
    if label not in SBS_OUTCOMES:
        raise ValueError(f"ideal_kraus_label must be one of {SBS_OUTCOMES}")
    z, x = label.removeprefix("K_")
    return [x, z]


def _xz_to_kraus(values: Sequence[str]) -> str:
    if len(values) != 2 or any(value not in {"g", "e"} for value in values):
        raise ValueError("values must contain X/Z g/e outcomes")
    return f"K_{values[1]}{values[0]}"


@dataclass(frozen=True)
class SBSFaultOverlayConfig:
    """sBs constituent × stage 的 ancilla flip/backaction assumptions。"""

    bit_flip_probabilities: tuple[tuple[float, float, float], tuple[float, float, float]]
    phase_flip_probabilities: tuple[tuple[float, float, float], tuple[float, float, float]]
    logical_fault_given_big_cd_bit: tuple[float, float]
    phase_backaction_scale: tuple[float, float]
    small_cd_bit_backaction_scale: tuple[float, float]
    misclassification_rotation_max_rad: float
    parameter_provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bit_flip_probabilities",
            _probability_matrix(
                self.bit_flip_probabilities,
                (2, 3),
                "bit_flip_probabilities",
            ),
        )
        object.__setattr__(
            self,
            "phase_flip_probabilities",
            _probability_matrix(
                self.phase_flip_probabilities,
                (2, 3),
                "phase_flip_probabilities",
            ),
        )
        object.__setattr__(
            self,
            "logical_fault_given_big_cd_bit",
            _probability_vector(
                self.logical_fault_given_big_cd_bit,
                2,
                "logical_fault_given_big_cd_bit",
            ),
        )
        for name in ("phase_backaction_scale", "small_cd_bit_backaction_scale"):
            try:
                pair = tuple(float(value) for value in getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain two finite non-negative values") from exc
            if len(pair) != 2 or not all(isfinite(value) and value >= 0.0 for value in pair):
                raise ValueError(f"{name} must contain two finite non-negative values")
            object.__setattr__(self, name, pair)
        rotation = float(self.misclassification_rotation_max_rad)
        if not isfinite(rotation) or rotation < 0.0:
            raise ValueError("misclassification_rotation_max_rad must be finite and non-negative")
        object.__setattr__(self, "misclassification_rotation_max_rad", rotation)
        object.__setattr__(
            self,
            "parameter_provenance",
            _text(self.parameter_provenance, "parameter_provenance"),
        )


@dataclass(frozen=True)
class SBSAncillaFaultEvent:
    constituent: str
    fault_type: str
    stage: str
    toggles_z_basis_outcome: bool
    logical_backaction: bool
    continuous_backaction: float


@dataclass(frozen=True)
class SBSFaultOverlayTruth:
    original_ideal_kraus_label: str
    faulted_ideal_kraus_label: str
    events: tuple[SBSAncillaFaultEvent, ...]
    logical_backaction_by_constituent: tuple[bool, bool]
    continuous_backaction_by_constituent: tuple[float, float]
    readout_misclassified: tuple[bool, bool]
    virtual_rotation_error_rad: tuple[float, float]
    parameter_provenance: str
    truth_scope: str = "simulator_fault_truth_not_deployable_input"


@dataclass(frozen=True)
class SBSFaultOverlayStep:
    observation_reset: SBSObservationResetStep
    fault_truth: SBSFaultOverlayTruth

    def deployable_record(self) -> dict[str, object]:
        return self.observation_reset.observed.as_deployable_dict()


@dataclass(frozen=True)
class SBSFaultOverlayTrajectory:
    steps: tuple[SBSFaultOverlayStep, ...]
    final_hidden_memory: HiddenAncillaMemory
    final_observed_memory: ObservedSyndromeMemory
    seed: int
    protocol_id: str = SBS_PROTOCOL_ID

    def deployable_records(self) -> tuple[dict[str, object], ...]:
        return tuple(step.deployable_record() for step in self.steps)


class SBSAncillaFaultOverlay:
    """在既有 sBs observation/reset kernel 上叠加 stage-resolved flips。"""

    protocol_id = SBS_PROTOCOL_ID
    device_calibrated = False

    def __init__(
        self,
        base_model: SBSObservationResetModel,
        config: SBSFaultOverlayConfig,
    ) -> None:
        if not isinstance(base_model, SBSObservationResetModel):
            raise TypeError("base_model must be an SBSObservationResetModel")
        if not isinstance(config, SBSFaultOverlayConfig):
            raise TypeError("config must be an SBSFaultOverlayConfig")
        if base_model.protocol_id != SBS_PROTOCOL_ID:
            raise ValueError("base_model must use the SBS main protocol ID")
        self.base_model = base_model
        self.config = config

    def _step_from_sequences(
        self,
        ideal_kraus_label: str,
        *,
        hidden_memory: HiddenAncillaMemory,
        observed_memory: ObservedSyndromeMemory,
        fault_sequence: np.random.SeedSequence,
        base_sequence: np.random.SeedSequence,
    ) -> SBSFaultOverlayStep:
        values = _kraus_to_xz(ideal_kraus_label)
        rng = np.random.default_rng(fault_sequence)
        events: list[SBSAncillaFaultEvent] = []
        logical = [False, False]
        continuous = [0.0, 0.0]

        for constituent_index, constituent in enumerate(SBS_CONSTITUENTS):
            toggle_count = 0
            for stage_index, stage in enumerate(SBS_FAULT_STAGES):
                bit = float(rng.random()) < self.config.bit_flip_probabilities[
                    constituent_index
                ][stage_index]
                if bit:
                    toggle_count += 1
                    is_logical = False
                    backaction = 0.0
                    if stage == "big_cd":
                        is_logical = float(rng.random()) < self.config.logical_fault_given_big_cd_bit[
                            constituent_index
                        ]
                    elif stage == "small_cd":
                        sign = -1.0 if float(rng.random()) < 0.5 else 1.0
                        backaction = sign * self.config.small_cd_bit_backaction_scale[
                            constituent_index
                        ]
                    logical[constituent_index] ^= is_logical
                    continuous[constituent_index] += backaction
                    events.append(
                        SBSAncillaFaultEvent(
                            constituent=constituent,
                            fault_type="bit_flip",
                            stage=stage,
                            toggles_z_basis_outcome=True,
                            logical_backaction=is_logical,
                            continuous_backaction=backaction,
                        )
                    )

                phase = float(rng.random()) < self.config.phase_flip_probabilities[
                    constituent_index
                ][stage_index]
                if phase:
                    # z-basis readout 不被 phase flip 翻转；SBS unitary 中只保留小的
                    # signed effective displacement/backaction assumption。
                    sign = -1.0 if float(rng.random()) < 0.5 else 1.0
                    backaction = (
                        0.0
                        if stage == "readout"
                        else sign * self.config.phase_backaction_scale[constituent_index]
                    )
                    continuous[constituent_index] += backaction
                    events.append(
                        SBSAncillaFaultEvent(
                            constituent=constituent,
                            fault_type="phase_flip",
                            stage=stage,
                            toggles_z_basis_outcome=False,
                            logical_backaction=False,
                            continuous_backaction=backaction,
                        )
                    )
            if toggle_count % 2:
                values[constituent_index] = _toggle_ge(values[constituent_index])

        faulted_label = _xz_to_kraus(values)
        base_seed = int(base_sequence.generate_state(1, dtype=np.uint64)[0])
        base = self.base_model.step(
            faulted_label,
            hidden_memory=hidden_memory,
            observed_memory=observed_memory,
            seed=base_seed,
        )
        observed = base.observed.syndrome.as_tuple()
        hidden = base.truth.hidden_pre_readout
        misclassified: list[bool] = []
        rotations: list[float] = []
        for hidden_value, observed_value in zip(hidden, observed):
            mismatch = hidden_value in {"g", "e"} and observed_value != hidden_value
            misclassified.append(mismatch)
            rotations.append(
                float(
                    rng.uniform(
                        -self.config.misclassification_rotation_max_rad,
                        self.config.misclassification_rotation_max_rad,
                    )
                )
                if mismatch
                else 0.0
            )
        truth = SBSFaultOverlayTruth(
            original_ideal_kraus_label=ideal_kraus_label,
            faulted_ideal_kraus_label=faulted_label,
            events=tuple(events),
            logical_backaction_by_constituent=(logical[0], logical[1]),
            continuous_backaction_by_constituent=(continuous[0], continuous[1]),
            readout_misclassified=(misclassified[0], misclassified[1]),
            virtual_rotation_error_rad=(rotations[0], rotations[1]),
            parameter_provenance=self.config.parameter_provenance,
        )
        return SBSFaultOverlayStep(observation_reset=base, fault_truth=truth)

    def step(
        self,
        ideal_kraus_label: str,
        *,
        hidden_memory: HiddenAncillaMemory | None = None,
        observed_memory: ObservedSyndromeMemory | None = None,
        seed: int,
    ) -> SBSFaultOverlayStep:
        normalized_seed = _seed(seed)
        fault_sequence, base_sequence = np.random.SeedSequence(normalized_seed).spawn(2)
        return self._step_from_sequences(
            ideal_kraus_label,
            hidden_memory=HiddenAncillaMemory() if hidden_memory is None else hidden_memory,
            observed_memory=(
                ObservedSyndromeMemory() if observed_memory is None else observed_memory
            ),
            fault_sequence=fault_sequence,
            base_sequence=base_sequence,
        )

    def simulate(
        self,
        ideal_kraus_labels: Sequence[str],
        *,
        seed: int,
        initial_hidden_memory: HiddenAncillaMemory | None = None,
        initial_observed_memory: ObservedSyndromeMemory | None = None,
    ) -> SBSFaultOverlayTrajectory:
        if isinstance(ideal_kraus_labels, (str, bytes)):
            raise TypeError("ideal_kraus_labels must be a sequence, not text")
        labels = tuple(ideal_kraus_labels)
        if any(label not in SBS_OUTCOMES for label in labels):
            raise ValueError(f"all labels must be in {SBS_OUTCOMES}")
        normalized_seed = _seed(seed)
        cycle_sequences = np.random.SeedSequence(normalized_seed).spawn(len(labels))
        hidden = HiddenAncillaMemory() if initial_hidden_memory is None else initial_hidden_memory
        observed = (
            ObservedSyndromeMemory()
            if initial_observed_memory is None
            else initial_observed_memory
        )
        steps: list[SBSFaultOverlayStep] = []
        for label, cycle_sequence in zip(labels, cycle_sequences):
            fault_sequence, base_sequence = cycle_sequence.spawn(2)
            step = self._step_from_sequences(
                label,
                hidden_memory=hidden,
                observed_memory=observed,
                fault_sequence=fault_sequence,
                base_sequence=base_sequence,
            )
            steps.append(step)
            hidden = step.observation_reset.hidden_memory
            observed = step.observation_reset.observed_memory
        return SBSFaultOverlayTrajectory(
            steps=tuple(steps),
            final_hidden_memory=hidden,
            final_observed_memory=observed,
            seed=normalized_seed,
        )


@dataclass(frozen=True)
class SharpenTrimAncillaConfig:
    """四轮 sharpen--trim 原生 ancilla/readout/reset assumptions。"""

    bit_flip_probabilities: tuple[float, float, float, float]
    phase_flip_probabilities: tuple[float, float, float, float]
    leakage_injection_probabilities: tuple[float, float, float, float]
    readout_confusion: ArrayLike
    correct_reset_success: tuple[float, float]
    wrong_sign_reset_success: float
    leakage_reset_success: float
    peak_feedback_fraction: float
    peak_feedback_asymmetry_fraction: float
    trim_feedback_fraction: float
    lattice: float
    counter_max: int
    parameter_provenance: str
    readout_provenance: str
    reset_provenance: str

    def __post_init__(self) -> None:
        for name in (
            "bit_flip_probabilities",
            "phase_flip_probabilities",
            "leakage_injection_probabilities",
        ):
            object.__setattr__(
                self,
                name,
                _probability_vector(getattr(self, name), 4, name),
            )
        object.__setattr__(
            self,
            "readout_confusion",
            _row_stochastic(self.readout_confusion, (3, 2), "readout_confusion"),
        )
        object.__setattr__(
            self,
            "correct_reset_success",
            _probability_vector(self.correct_reset_success, 2, "correct_reset_success"),
        )
        object.__setattr__(
            self,
            "wrong_sign_reset_success",
            _probability(self.wrong_sign_reset_success, "wrong_sign_reset_success"),
        )
        object.__setattr__(
            self,
            "leakage_reset_success",
            _probability(self.leakage_reset_success, "leakage_reset_success"),
        )
        for name in (
            "peak_feedback_fraction",
            "trim_feedback_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        asymmetry = float(self.peak_feedback_asymmetry_fraction)
        if not isfinite(asymmetry):
            raise ValueError("peak_feedback_asymmetry_fraction must be finite")
        object.__setattr__(self, "peak_feedback_asymmetry_fraction", asymmetry)
        lattice = float(self.lattice)
        if not isfinite(lattice) or lattice <= 0.0:
            raise ValueError("lattice must be finite and positive")
        object.__setattr__(self, "lattice", lattice)
        maximum = _counter(self.counter_max, "counter_max")
        if maximum < 1:
            raise ValueError("counter_max must be at least 1")
        object.__setattr__(self, "counter_max", maximum)
        for name in (
            "parameter_provenance",
            "readout_provenance",
            "reset_provenance",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))


@dataclass(frozen=True)
class SharpenTrimMemory:
    carry_state: str = "g"
    round_index: int = 0
    leakage_run: int = 0
    reset_failure_run: int = 0
    pauli_frame_x: int = 0
    pauli_frame_z: int = 0

    def __post_init__(self) -> None:
        if self.carry_state not in SHARPEN_TRIM_CARRY_STATES:
            raise ValueError(f"carry_state must be one of {SHARPEN_TRIM_CARRY_STATES}")
        for name in ("round_index", "leakage_run", "reset_failure_run"):
            object.__setattr__(self, name, _counter(getattr(self, name), name))
        for name in ("pauli_frame_x", "pauli_frame_z"):
            value = _counter(getattr(self, name), name)
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SharpenTrimObservedRound:
    round_index: int
    round_type: str
    observed_y: str
    feedback_axis: str
    feedback_displacement: float
    reset_action: str
    pauli_frame_x: int
    pauli_frame_z: int
    protocol_id: str = SHARPEN_TRIM_PROTOCOL_ID
    observation_scope: str = "native_binary_sigma_y_observation"

    def as_deployable_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "round_type": self.round_type,
            "observed_y": self.observed_y,
            "feedback_axis": self.feedback_axis,
            "feedback_displacement": self.feedback_displacement,
            "reset_action": self.reset_action,
            "pauli_frame_x": self.pauli_frame_x,
            "pauli_frame_z": self.pauli_frame_z,
            "protocol_id": self.protocol_id,
            "observation_scope": self.observation_scope,
        }


@dataclass(frozen=True)
class SharpenTrimTruthRound:
    round_index: int
    round_type: str
    ideal_y: str
    hidden_pre_readout: str
    hidden_post_reset: str
    bit_flip: bool
    bit_flip_interaction_fraction: float | None
    phase_flip: bool
    leakage_injected: bool
    readout_misclassified: bool
    feedback_direction_wrong: bool
    logical_backaction: str
    reset_succeeded: bool
    truth_scope: str = "simulator_sharpen_trim_truth_not_deployable_input"


@dataclass(frozen=True)
class SharpenTrimStep:
    observed: SharpenTrimObservedRound
    truth: SharpenTrimTruthRound
    memory: SharpenTrimMemory


@dataclass(frozen=True)
class SharpenTrimTrajectory:
    observed_rounds: tuple[SharpenTrimObservedRound, ...]
    truth_rounds: tuple[SharpenTrimTruthRound, ...]
    final_memory: SharpenTrimMemory
    seed: int
    protocol_id: str = SHARPEN_TRIM_PROTOCOL_ID

    def deployable_records(self) -> tuple[dict[str, object], ...]:
        return tuple(item.as_deployable_dict() for item in self.observed_rounds)


def _intrinsic_frame_flip(round_type: str) -> tuple[int, int]:
    # Campagne square-code convention: q-peak and p-trim carry Z; p-peak and
    # q-trim carry X. Sign +/- gives the same Pauli modulo stabilizers.
    if round_type in {"p_peak_sharpen", "q_envelope_trim"}:
        return 1, 0
    if round_type in {"q_peak_sharpen", "p_envelope_trim"}:
        return 0, 1
    raise ValueError("unknown sharpen-trim round type")


def _bit_flip_logical_backaction(round_type: str, fraction: float | None) -> str:
    if fraction is None or not 0.25 <= fraction <= 0.75:
        return "I"
    if round_type == "p_peak_sharpen":
        return "X"
    if round_type == "q_peak_sharpen":
        return "Z"
    return "I"


class SharpenTrimAncillaModel:
    """四轮 ``+y/-y`` measurement-feedback ancilla state machine。"""

    protocol_id = SHARPEN_TRIM_PROTOCOL_ID
    device_calibrated = False

    def __init__(self, config: SharpenTrimAncillaConfig) -> None:
        if not isinstance(config, SharpenTrimAncillaConfig):
            raise TypeError("config must be a SharpenTrimAncillaConfig")
        self.config = config

    def _saturating_increment(self, value: int) -> int:
        return min(self.config.counter_max, value + 1)

    def _step_with_rng(
        self,
        ideal_y: str,
        *,
        memory: SharpenTrimMemory,
        rng: np.random.Generator,
    ) -> SharpenTrimStep:
        if ideal_y not in SHARPEN_TRIM_OBSERVED_CLASSES:
            raise ValueError("ideal_y must be +y or -y")
        if not isinstance(memory, SharpenTrimMemory):
            raise TypeError("memory must be a SharpenTrimMemory")
        round_type = SHARPEN_TRIM_ROUND_TYPES[
            memory.round_index % len(SHARPEN_TRIM_ROUND_TYPES)
        ]
        type_index = SHARPEN_TRIM_ROUND_TYPES.index(round_type)

        hidden = ideal_y if memory.carry_state == "g" else memory.carry_state
        if hidden not in SHARPEN_TRIM_HIDDEN_STATES:
            raise RuntimeError("invalid sharpen-trim hidden carry")
        leakage_injected = (
            float(rng.random()) < self.config.leakage_injection_probabilities[type_index]
        )
        if leakage_injected:
            hidden = "leakage"

        bit_flip = float(rng.random()) < self.config.bit_flip_probabilities[type_index]
        phase_flip = float(rng.random()) < self.config.phase_flip_probabilities[type_index]
        bit_fraction = float(rng.random()) if bit_flip else None
        if hidden in {"+y", "-y"}:
            if bit_flip:
                hidden = _toggle_y(hidden)
            if phase_flip:
                hidden = _toggle_y(hidden)

        hidden_index = SHARPEN_TRIM_HIDDEN_STATES.index(hidden)
        observed_index = int(rng.choice(2, p=self.config.readout_confusion[hidden_index]))
        observed = SHARPEN_TRIM_OBSERVED_CLASSES[observed_index]
        readout_misclassified = hidden in {"+y", "-y"} and observed != hidden
        feedback_wrong = observed != ideal_y

        if "peak" in round_type:
            sign = 1.0 if observed == "+y" else -1.0
            displacement = self.config.lattice * (
                sign * self.config.peak_feedback_fraction
                + self.config.peak_feedback_asymmetry_fraction
            )
        else:
            sign = 1.0 if observed == "+y" else -1.0
            displacement = sign * self.config.lattice * self.config.trim_feedback_fraction
        axis = "q" if round_type.startswith("q_") else "p"

        reset_action = (
            "conditional_pi_over_2_reset_from_plus_y"
            if observed == "+y"
            else "conditional_pi_over_2_reset_from_minus_y"
        )
        if hidden == "leakage":
            reset_probability = self.config.leakage_reset_success
        elif observed == hidden:
            reset_probability = self.config.correct_reset_success[observed_index]
        else:
            reset_probability = self.config.wrong_sign_reset_success
        reset_succeeded = float(rng.random()) < reset_probability
        post = "g" if reset_succeeded else hidden

        leakage_run = (
            self._saturating_increment(memory.leakage_run)
            if hidden == "leakage"
            else 0
        )
        reset_failure_run = (
            0
            if reset_succeeded
            else self._saturating_increment(memory.reset_failure_run)
        )
        frame_x, frame_z = _intrinsic_frame_flip(round_type)
        logical_backaction = _bit_flip_logical_backaction(round_type, bit_fraction)
        # The deployable frame may include only deterministic protocol updates.
        # Stochastic physical logical backaction remains simulator truth because
        # the controller does not observe whether the mid-interaction bit flip
        # occurred.
        next_frame_x = memory.pauli_frame_x ^ frame_x
        next_frame_z = memory.pauli_frame_z ^ frame_z
        next_memory = SharpenTrimMemory(
            carry_state=post,
            round_index=memory.round_index + 1,
            leakage_run=leakage_run,
            reset_failure_run=reset_failure_run,
            pauli_frame_x=next_frame_x,
            pauli_frame_z=next_frame_z,
        )
        observed_round = SharpenTrimObservedRound(
            round_index=memory.round_index,
            round_type=round_type,
            observed_y=observed,
            feedback_axis=axis,
            feedback_displacement=displacement,
            reset_action=reset_action,
            pauli_frame_x=next_frame_x,
            pauli_frame_z=next_frame_z,
        )
        truth = SharpenTrimTruthRound(
            round_index=memory.round_index,
            round_type=round_type,
            ideal_y=ideal_y,
            hidden_pre_readout=hidden,
            hidden_post_reset=post,
            bit_flip=bit_flip,
            bit_flip_interaction_fraction=bit_fraction,
            phase_flip=phase_flip,
            leakage_injected=leakage_injected,
            readout_misclassified=readout_misclassified,
            feedback_direction_wrong=feedback_wrong,
            logical_backaction=logical_backaction,
            reset_succeeded=reset_succeeded,
        )
        return SharpenTrimStep(observed=observed_round, truth=truth, memory=next_memory)

    def step(
        self,
        ideal_y: str,
        *,
        memory: SharpenTrimMemory | None = None,
        seed: int,
    ) -> SharpenTrimStep:
        return self._step_with_rng(
            ideal_y,
            memory=SharpenTrimMemory() if memory is None else memory,
            rng=np.random.default_rng(_seed(seed)),
        )

    def simulate(
        self,
        ideal_y_outcomes: Sequence[str],
        *,
        seed: int,
        initial_memory: SharpenTrimMemory | None = None,
    ) -> SharpenTrimTrajectory:
        if isinstance(ideal_y_outcomes, (str, bytes)):
            raise TypeError("ideal_y_outcomes must be a sequence, not text")
        outcomes = tuple(ideal_y_outcomes)
        if any(value not in SHARPEN_TRIM_OBSERVED_CLASSES for value in outcomes):
            raise ValueError("all ideal outcomes must be +y or -y")
        normalized_seed = _seed(seed)
        rng = np.random.default_rng(normalized_seed)
        memory = SharpenTrimMemory() if initial_memory is None else initial_memory
        observed: list[SharpenTrimObservedRound] = []
        truth: list[SharpenTrimTruthRound] = []
        for outcome in outcomes:
            step = self._step_with_rng(outcome, memory=memory, rng=rng)
            observed.append(step.observed)
            truth.append(step.truth)
            memory = step.memory
        return SharpenTrimTrajectory(
            observed_rounds=tuple(observed),
            truth_rounds=tuple(truth),
            final_memory=memory,
            seed=normalized_seed,
        )


@dataclass(frozen=True)
class SecondaryProtocolNoiseSpec:
    protocol_id: str
    role: str
    executable: bool
    allowed_scan_parameters: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    primary_source_required: bool = True


SECONDARY_PROTOCOL_NOISE_REGISTRY: Mapping[str, SecondaryProtocolNoiseSpec] = MappingProxyType(
    {
        "steane": SecondaryProtocolNoiseSpec(
            protocol_id="PROTO-STEANE-THEORY-ONLY",
            role="theoretical_comparator",
            executable=False,
            allowed_scan_parameters=("data_variance", "ancilla_variance"),
            forbidden_claims=("SBS main ranking", "hardware implementation", "device timing"),
        ),
        "knill_qunaught": SecondaryProtocolNoiseSpec(
            protocol_id="PROTO-KNILL-QUNAUGHT-SECONDARY",
            role="secondary_noise_shaping_comparator",
            executable=False,
            allowed_scan_parameters=("resource_squeezing", "homodyne_variance"),
            forbidden_claims=("SBS main ranking", "FPGA physical control", "verified 1e-8"),
        ),
        "p_steane": SecondaryProtocolNoiseSpec(
            protocol_id="PROTO-ME-PSTEANE-SECONDARY",
            role="secondary_tunable_preprocessing_comparator",
            executable=False,
            allowed_scan_parameters=("a", "b", "data_to_ancilla_noise_ratio"),
            forbidden_claims=("FPGA implements squeezing", "project-verified optimum", "device timing"),
        ),
    }
)


def secondary_protocol_noise_specs() -> tuple[SecondaryProtocolNoiseSpec, ...]:
    return tuple(SECONDARY_PROTOCOL_NOISE_REGISTRY.values())


@dataclass(frozen=True)
class ProtocolAncillaValidationResult:
    samples: int
    seed: int
    sbs_expected_big_cd_bit_rate: float
    sbs_observed_big_cd_bit_rate: float
    sbs_expected_logical_backaction_rate: float
    sbs_observed_logical_backaction_rate: float
    sbs_phase_outcome_toggle_rate: float
    sharpen_expected_feedback_mismatch_rate: float
    sharpen_observed_feedback_mismatch_rate: float
    sharpen_bit_middle_window_rate: float
    sharpen_bit_logical_backaction_rate: float
    checks: Mapping[str, bool]
    evidence_scope: str = "protocol_native_ancilla_error_effective_validation_not_device_calibrated"

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "sbs": {
                "expected_big_cd_bit_rate": self.sbs_expected_big_cd_bit_rate,
                "observed_big_cd_bit_rate": self.sbs_observed_big_cd_bit_rate,
                "expected_logical_backaction_rate": self.sbs_expected_logical_backaction_rate,
                "observed_logical_backaction_rate": self.sbs_observed_logical_backaction_rate,
                "phase_outcome_toggle_rate": self.sbs_phase_outcome_toggle_rate,
            },
            "sharpen_trim": {
                "expected_feedback_mismatch_rate": self.sharpen_expected_feedback_mismatch_rate,
                "observed_feedback_mismatch_rate": self.sharpen_observed_feedback_mismatch_rate,
                "bit_middle_window_rate": self.sharpen_bit_middle_window_rate,
                "bit_logical_backaction_rate": self.sharpen_bit_logical_backaction_rate,
            },
            "checks": {name: bool(value) for name, value in self.checks.items()},
            "secondary_protocols": [
                {
                    "protocol_id": spec.protocol_id,
                    "role": spec.role,
                    "executable": spec.executable,
                    "allowed_scan_parameters": list(spec.allowed_scan_parameters),
                    "forbidden_claims": list(spec.forbidden_claims),
                }
                for spec in secondary_protocol_noise_specs()
            ],
            "evidence_scope": self.evidence_scope,
            "claim_boundary": {
                "allowed": "protocol-native effective ancilla/readout/reset fault flow",
                "forbidden": "device-calibrated fault rates, master-equation fidelity, or secondary-protocol hardware execution",
            },
        }


def _perfect_sbs_base_model() -> SBSObservationResetModel:
    return make_persistent_leakage_model(
        readout_confusion=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=1.0,
        counter_max=2**31 - 1,
        readout_provenance="T2.2.2 validation perfect classifier assumption",
        parameter_provenance="T2.2.2 validation no-leakage perfect-reset assumption",
    )


def run_protocol_ancilla_validation(
    *,
    samples: int = 60_000,
    seed: int = 2026071422,
) -> ProtocolAncillaValidationResult:
    if not isinstance(samples, int) or samples < 10_000:
        raise ValueError("samples must be an integer >= 10000")
    normalized_seed = _seed(seed)
    bit_rate = 0.02
    phase_rate = 0.03
    logical_given_bit = 0.5
    sbs = SBSAncillaFaultOverlay(
        _perfect_sbs_base_model(),
        SBSFaultOverlayConfig(
            bit_flip_probabilities=((0.0, bit_rate, 0.0), (0.0, 0.0, 0.0)),
            phase_flip_probabilities=((0.0, 0.0, 0.0), (0.0, phase_rate, 0.0)),
            logical_fault_given_big_cd_bit=(logical_given_bit, 0.0),
            phase_backaction_scale=(0.01, 0.01),
            small_cd_bit_backaction_scale=(0.02, 0.02),
            misclassification_rotation_max_rad=0.6,
            parameter_provenance="T2.2.2 seeded validation assumptions",
        ),
    )
    sbs_trajectory = sbs.simulate(("K_gg",) * samples, seed=normalized_seed)
    big_bit = np.fromiter(
        (
            any(
                event.constituent == "X"
                and event.fault_type == "bit_flip"
                and event.stage == "big_cd"
                for event in step.fault_truth.events
            )
            for step in sbs_trajectory.steps
        ),
        dtype=bool,
        count=samples,
    )
    logical = np.fromiter(
        (step.fault_truth.logical_backaction_by_constituent[0] for step in sbs_trajectory.steps),
        dtype=bool,
        count=samples,
    )
    phase_toggle = np.fromiter(
        (
            any(
                event.fault_type == "phase_flip"
                and event.toggles_z_basis_outcome
                for event in step.fault_truth.events
            )
            for step in sbs_trajectory.steps
        ),
        dtype=bool,
        count=samples,
    )

    phase_probability = 0.04
    readout_error = 0.015
    bit_probability = 0.02
    sharp = SharpenTrimAncillaModel(
        SharpenTrimAncillaConfig(
            bit_flip_probabilities=(bit_probability,) * 4,
            phase_flip_probabilities=(phase_probability,) * 4,
            leakage_injection_probabilities=(0.0,) * 4,
            readout_confusion=np.asarray(
                [
                    [1.0 - readout_error, readout_error],
                    [readout_error, 1.0 - readout_error],
                    [0.5, 0.5],
                ]
            ),
            correct_reset_success=(1.0, 1.0),
            wrong_sign_reset_success=1.0,
            leakage_reset_success=0.0,
            peak_feedback_fraction=0.08,
            peak_feedback_asymmetry_fraction=0.0,
            trim_feedback_fraction=0.5,
            lattice=LATTICE_CONST,
            counter_max=2**31 - 1,
            parameter_provenance="T2.2.2 seeded validation assumptions",
            readout_provenance="T2.2.2 symmetric binary confusion assumption",
            reset_provenance="T2.2.2 perfect reset isolation assumption",
        )
    )
    sharp_trajectory = sharp.simulate(("+y",) * samples, seed=normalized_seed + 1)
    mismatch = np.fromiter(
        (item.feedback_direction_wrong for item in sharp_trajectory.truth_rounds),
        dtype=bool,
        count=samples,
    )
    bit_middle = np.fromiter(
        (
            item.bit_flip
            and item.bit_flip_interaction_fraction is not None
            and 0.25 <= item.bit_flip_interaction_fraction <= 0.75
            for item in sharp_trajectory.truth_rounds
        ),
        dtype=bool,
        count=samples,
    )
    bit_logical = np.fromiter(
        (item.logical_backaction != "I" for item in sharp_trajectory.truth_rounds),
        dtype=bool,
        count=samples,
    )
    # bit 与 phase 都翻转 Y outcome；两者同时发生会抵消。随后再经过 symmetric
    # binary readout confusion。
    toggle_probability = (
        bit_probability * (1.0 - phase_probability)
        + (1.0 - bit_probability) * phase_probability
    )
    expected_mismatch = (
        toggle_probability * (1.0 - readout_error)
        + (1.0 - toggle_probability) * readout_error
    )
    sbs_bit_observed = float(np.mean(big_bit))
    sbs_logical_observed = float(np.mean(logical))
    sharp_mismatch_observed = float(np.mean(mismatch))
    standard_error_bit = np.sqrt(bit_rate * (1.0 - bit_rate) / samples)
    standard_error_logical = np.sqrt(
        bit_rate * logical_given_bit * (1.0 - bit_rate * logical_given_bit) / samples
    )
    standard_error_mismatch = np.sqrt(
        expected_mismatch * (1.0 - expected_mismatch) / samples
    )
    checks = MappingProxyType(
        {
            "sbs_big_cd_bit_rate_within_5se": abs(sbs_bit_observed - bit_rate)
            <= 5.0 * standard_error_bit,
            "sbs_logical_backaction_within_5se": abs(
                sbs_logical_observed - bit_rate * logical_given_bit
            )
            <= 5.0 * standard_error_logical,
            "sbs_phase_does_not_toggle_z_basis": not bool(np.any(phase_toggle)),
            "sharpen_feedback_mismatch_within_5se": abs(
                sharp_mismatch_observed - expected_mismatch
            )
            <= 5.0 * standard_error_mismatch,
            "sharpen_bit_middle_window_is_half": abs(
                float(np.mean(bit_middle)) - 0.5 * bit_probability
            )
            <= 5.0 * standard_error_logical,
            "secondary_protocols_remain_non_executable": all(
                not spec.executable for spec in secondary_protocol_noise_specs()
            ),
        }
    )
    return ProtocolAncillaValidationResult(
        samples=samples,
        seed=normalized_seed,
        sbs_expected_big_cd_bit_rate=bit_rate,
        sbs_observed_big_cd_bit_rate=sbs_bit_observed,
        sbs_expected_logical_backaction_rate=bit_rate * logical_given_bit,
        sbs_observed_logical_backaction_rate=sbs_logical_observed,
        sbs_phase_outcome_toggle_rate=float(np.mean(phase_toggle)),
        sharpen_expected_feedback_mismatch_rate=expected_mismatch,
        sharpen_observed_feedback_mismatch_rate=sharp_mismatch_observed,
        sharpen_bit_middle_window_rate=float(np.mean(bit_middle)),
        sharpen_bit_logical_backaction_rate=float(np.mean(bit_logical)),
        checks=checks,
    )


def write_protocol_ancilla_validation(
    result: ProtocolAncillaValidationResult,
    output_path: str | Path,
) -> Path:
    if not isinstance(result, ProtocolAncillaValidationResult):
        raise TypeError("result must be a ProtocolAncillaValidationResult")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=2026071422)
    arguments = parser.parse_args()
    result = run_protocol_ancilla_validation(samples=arguments.samples, seed=arguments.seed)
    write_protocol_ancilla_validation(result, arguments.output)
    print(json.dumps(result.as_dict()["checks"], ensure_ascii=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "SBS_FAULT_STAGES",
    "SBS_CONSTITUENTS",
    "SHARPEN_TRIM_PROTOCOL_ID",
    "SHARPEN_TRIM_ROUND_TYPES",
    "SHARPEN_TRIM_HIDDEN_STATES",
    "SHARPEN_TRIM_OBSERVED_CLASSES",
    "SBSFaultOverlayConfig",
    "SBSAncillaFaultEvent",
    "SBSFaultOverlayTruth",
    "SBSFaultOverlayStep",
    "SBSFaultOverlayTrajectory",
    "SBSAncillaFaultOverlay",
    "SharpenTrimAncillaConfig",
    "SharpenTrimMemory",
    "SharpenTrimObservedRound",
    "SharpenTrimTruthRound",
    "SharpenTrimStep",
    "SharpenTrimTrajectory",
    "SharpenTrimAncillaModel",
    "SecondaryProtocolNoiseSpec",
    "SECONDARY_PROTOCOL_NOISE_REGISTRY",
    "secondary_protocol_noise_specs",
    "ProtocolAncillaValidationResult",
    "run_protocol_ancilla_validation",
    "write_protocol_ancilla_validation",
]
