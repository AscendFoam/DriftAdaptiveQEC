"""Native sharpen-trim ancilla, readout, reset, and hidden-carry model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .common import (
    SHARPEN_TRIM_CARRY_STATES,
    SHARPEN_TRIM_HIDDEN_STATES,
    SHARPEN_TRIM_OBSERVED_CLASSES,
    SHARPEN_TRIM_PROTOCOL_ID,
    SHARPEN_TRIM_ROUND_TYPES,
    _counter,
    _probability,
    _probability_vector,
    _row_stochastic,
    _seed,
    _text,
)

def _toggle_y(value: str) -> str:
    if value == "+y":
        return "-y"
    if value == "-y":
        return "+y"
    raise ValueError("Y-basis value must be +y or -y")

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

