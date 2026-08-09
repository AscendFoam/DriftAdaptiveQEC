"""Stage-resolved SBS ancilla fault overlay."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from ..sbs_error_space import SBS_OUTCOMES, SBS_PROTOCOL_ID
from ..sbs_observation_reset import (
    HiddenAncillaMemory,
    ObservedSyndromeMemory,
    SBSObservationResetModel,
    SBSObservationResetStep,
)
from .common import (
    SBS_CONSTITUENTS,
    SBS_FAULT_STAGES,
    _probability_matrix,
    _probability_vector,
    _seed,
    _text,
)

def _toggle_ge(value: str) -> str:
    if value == "g":
        return "e"
    if value == "e":
        return "g"
    raise ValueError("sBs ideal value must be g or e")

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

