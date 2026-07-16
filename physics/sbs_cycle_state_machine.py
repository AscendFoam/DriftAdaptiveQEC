"""Sivak Table S3 对齐的 sBs constituent/full-XZ cycle state machine。

这里的 ns 只表示文献最高增益实验电路的 reference timeline。state machine 不调用
sleep、不测量软件耗时，也不把 4.924 us 写成目标低价 FPGA 的实测 latency。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .sbs_error_space import PauliFrame, SBS_PROTOCOL_ID
from .sbs_observation_reset import OBSERVED_CLASSES, SBSObservedCycle


TABLE_S3_PROFILE_ID = "SIVAK-2023-TABLE-S3-HIGHEST-GAIN-CONSTITUENT"
TIMING_EVIDENCE_SCOPE = "literature_reference_not_target_board_measurement"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


@dataclass(frozen=True)
class CyclePhaseSpec:
    phase_id: str
    group: str
    duration_ns: int
    literature_component: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_id", _text(self.phase_id, "phase_id"))
        object.__setattr__(self, "group", _text(self.group, "group"))
        object.__setattr__(
            self,
            "duration_ns",
            _nonnegative_integer(self.duration_ns, "duration_ns"),
        )
        if self.duration_ns == 0:
            raise ValueError("duration_ns must be positive")
        object.__setattr__(
            self,
            "literature_component",
            _text(self.literature_component, "literature_component"),
        )


@dataclass(frozen=True)
class SBSCycleTimingProfile:
    profile_id: str
    phases: tuple[CyclePhaseSpec, ...]
    source_path: str
    source_line: int
    source_fragment: str
    prose_sbs_ns: int
    prose_reset_ns: int
    evidence_scope: str = TIMING_EVIDENCE_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id"))
        if not self.phases:
            raise ValueError("timing profile must contain phases")
        phase_ids = [phase.phase_id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("phase_id values must be unique")
        object.__setattr__(self, "source_path", _text(self.source_path, "source_path"))
        object.__setattr__(self, "source_line", _nonnegative_integer(self.source_line, "source_line"))
        if self.source_line < 1:
            raise ValueError("source_line must be at least 1")
        object.__setattr__(
            self,
            "source_fragment",
            _text(self.source_fragment, "source_fragment"),
        )
        object.__setattr__(self, "prose_sbs_ns", _nonnegative_integer(self.prose_sbs_ns, "prose_sbs_ns"))
        object.__setattr__(
            self,
            "prose_reset_ns",
            _nonnegative_integer(self.prose_reset_ns, "prose_reset_ns"),
        )
        if self.evidence_scope != TIMING_EVIDENCE_SCOPE:
            raise ValueError(f"evidence_scope must be {TIMING_EVIDENCE_SCOPE!r}")

    @property
    def total_duration_ns(self) -> int:
        return sum(phase.duration_ns for phase in self.phases)

    def group_duration_ns(self, group: str) -> int:
        return sum(phase.duration_ns for phase in self.phases if phase.group == group)

    @property
    def sbs_layer_sum_ns(self) -> int:
        return sum(
            phase.duration_ns for phase in self.phases if phase.phase_id.startswith("sbs_layer_")
        )

    @property
    def table_sbs_block_ns(self) -> int:
        return self.group_duration_ns("sbs")

    @property
    def table_reset_block_ns(self) -> int:
        return self.group_duration_ns("reset")


def sivak_table_s3_profile() -> SBSCycleTimingProfile:
    """返回逐项转录的 Table S3 最高增益 constituent timing profile。"""

    raw = (
        ("enter_cycle", "cycle_overhead", 24, "Enter cycle"),
        ("enter_sbs", "sbs", 24, "Enter SBS"),
        ("sbs_layer_1", "sbs", 502, "Circuit layer1"),
        ("sbs_layer_2", "sbs", 708, "Circuit layer2"),
        ("sbs_layer_3", "sbs", 262, "Circuit layer3"),
        ("sbs_layer_4", "sbs", 76, "Circuit layer4"),
        ("exit_sbs", "sbs", 24, "Exit SBS"),
        ("enter_reset", "reset", 24, "Enter reset"),
        ("roundtrip_delay", "reset", 300, "Roundtrip delay"),
        ("readout_acquisition", "reset", 1400, "Acquisition"),
        ("signal_processing", "reset", 332, "Signal processing"),
        ("syndrome_distribution", "reset", 100, "Distribution of s0 and s1"),
        ("branch_and_feedback", "reset", 200, "Branching and feedback"),
        ("exit_reset", "reset", 24, "Exit reset"),
        ("mixer_matrix_calculation", "virtual_rotation", 400, "Mixer matrix calculation"),
        ("mixer_update", "virtual_rotation", 48, "Mixer update"),
        ("idle", "idle", 452, "Idle delay"),
        ("exit_cycle", "cycle_overhead", 24, "Exit cycle"),
    )
    return SBSCycleTimingProfile(
        profile_id=TABLE_S3_PROFILE_ID,
        phases=tuple(CyclePhaseSpec(*entry) for entry in raw),
        source_path=(
            "relative_papers/Real-time_quantum_error_correction_beyond_break-even/"
            "Real-time_quantum_error_correction_beyond_break-even.md"
        ),
        source_line=903,
        source_fragment="Mixer matrix calculation",
        prose_sbs_ns=1546,
        prose_reset_ns=2332,
    )


@dataclass(frozen=True)
class SBSConstituentControl:
    quadrature: str
    observed_class: str
    reset_action: str
    virtual_rotation_key: str

    def __post_init__(self) -> None:
        if self.quadrature not in {"X", "Z"}:
            raise ValueError("quadrature must be X or Z")
        if self.observed_class not in OBSERVED_CLASSES:
            raise ValueError(f"observed_class must be one of {OBSERVED_CLASSES}")
        object.__setattr__(self, "reset_action", _text(self.reset_action, "reset_action"))
        object.__setattr__(
            self,
            "virtual_rotation_key",
            _text(self.virtual_rotation_key, "virtual_rotation_key"),
        )


@dataclass(frozen=True)
class CycleEvent:
    phase_id: str
    group: str
    start_ns: int
    end_ns: int
    duration_ns: int
    metadata: Mapping[str, object]
    timing_scope: str = TIMING_EVIDENCE_SCOPE


@dataclass(frozen=True)
class SBSConstituentCycleTrace:
    control: SBSConstituentControl
    events: tuple[CycleEvent, ...]
    start_ns: int
    end_ns: int
    total_duration_ns: int
    input_frame: PauliFrame
    output_frame: PauliFrame
    profile_id: str
    timing_scope: str = TIMING_EVIDENCE_SCOPE
    target_hardware_measured: bool = False


@dataclass(frozen=True)
class SBSFullXZCycleTrace:
    cycle_index: int
    x_constituent: SBSConstituentCycleTrace
    z_constituent: SBSConstituentCycleTrace
    start_ns: int
    end_ns: int
    total_duration_ns: int
    input_frame: PauliFrame
    output_frame: PauliFrame
    protocol_id: str = SBS_PROTOCOL_ID
    timing_scope: str = TIMING_EVIDENCE_SCOPE
    target_hardware_measured: bool = False


class SBSConstituentRuntime:
    """可逐 event 推进且拒绝非法 transition 的 constituent FSM runtime。"""

    def __init__(
        self,
        *,
        profile: SBSCycleTimingProfile,
        control: SBSConstituentControl,
        start_ns: int,
        input_frame: PauliFrame,
        virtual_rotation_provenance: str,
    ) -> None:
        self._profile = profile
        self._control = control
        self._start_ns = _nonnegative_integer(start_ns, "start_ns")
        if not isinstance(input_frame, PauliFrame):
            raise TypeError("input_frame must be a PauliFrame")
        self._input_frame = input_frame
        self._virtual_rotation_provenance = _text(
            virtual_rotation_provenance, "virtual_rotation_provenance"
        )
        self._next_index = 0
        self._events: list[CycleEvent] = []

    @property
    def is_complete(self) -> bool:
        return self._next_index == len(self._profile.phases)

    @property
    def next_phase_id(self) -> str | None:
        if self.is_complete:
            return None
        return self._profile.phases[self._next_index].phase_id

    def _event_metadata(self, phase_id: str) -> Mapping[str, object]:
        metadata: dict[str, object] = {"quadrature": self._control.quadrature}
        if phase_id == "signal_processing":
            metadata["classified_observation"] = self._control.observed_class
            metadata["classification_scope"] = "T2.0.3 observed class"
        elif phase_id == "syndrome_distribution":
            metadata["distributed_observation"] = self._control.observed_class
        elif phase_id == "branch_and_feedback":
            metadata["branch_observation"] = self._control.observed_class
            metadata["reset_action"] = self._control.reset_action
        elif phase_id in {"mixer_matrix_calculation", "mixer_update"}:
            metadata["virtual_rotation_key"] = self._control.virtual_rotation_key
            metadata["virtual_rotation_provenance"] = self._virtual_rotation_provenance
            metadata["sbs_quadrature_switch_rad"] = float(np.pi / 2.0)
        return MappingProxyType(metadata)

    def advance(self) -> CycleEvent:
        if self.is_complete:
            raise RuntimeError("constituent runtime is already complete")
        phase = self._profile.phases[self._next_index]
        start = self._start_ns + sum(event.duration_ns for event in self._events)
        end = start + phase.duration_ns
        event = CycleEvent(
            phase_id=phase.phase_id,
            group=phase.group,
            start_ns=start,
            end_ns=end,
            duration_ns=phase.duration_ns,
            metadata=self._event_metadata(phase.phase_id),
        )
        self._events.append(event)
        self._next_index += 1
        return event

    @property
    def trace(self) -> SBSConstituentCycleTrace:
        if not self.is_complete:
            raise RuntimeError("constituent runtime must complete before trace is available")
        output_frame = (
            self._input_frame.after_x_constituent()
            if self._control.quadrature == "X"
            else self._input_frame.after_z_constituent()
        )
        return SBSConstituentCycleTrace(
            control=self._control,
            events=tuple(self._events),
            start_ns=self._start_ns,
            end_ns=self._start_ns + self._profile.total_duration_ns,
            total_duration_ns=self._profile.total_duration_ns,
            input_frame=self._input_frame,
            output_frame=output_frame,
            profile_id=self._profile.profile_id,
        )

    def run_to_completion(self) -> SBSConstituentCycleTrace:
        while not self.is_complete:
            self.advance()
        return self.trace


class SBSCycleStateMachine:
    """Table-S3 constituent FSM 和 chronological X→Z full-cycle composer。"""

    protocol_id = SBS_PROTOCOL_ID

    def __init__(
        self,
        *,
        timing_profile: SBSCycleTimingProfile | None = None,
        virtual_rotation_key_by_observation: Mapping[str, str],
        virtual_rotation_provenance: str,
    ) -> None:
        self.timing_profile = sivak_table_s3_profile() if timing_profile is None else timing_profile
        if not isinstance(self.timing_profile, SBSCycleTimingProfile):
            raise TypeError("timing_profile must be an SBSCycleTimingProfile")
        if set(virtual_rotation_key_by_observation) != set(OBSERVED_CLASSES):
            raise ValueError(
                f"virtual_rotation_key_by_observation must have exactly {OBSERVED_CLASSES}"
            )
        keys = {
            observed: _text(virtual_rotation_key_by_observation[observed], f"vr_key[{observed}]")
            for observed in OBSERVED_CLASSES
        }
        self.virtual_rotation_key_by_observation = MappingProxyType(keys)
        self.virtual_rotation_provenance = _text(
            virtual_rotation_provenance, "virtual_rotation_provenance"
        )

    def start_constituent(
        self,
        control: SBSConstituentControl,
        *,
        start_ns: int = 0,
        input_frame: PauliFrame | None = None,
    ) -> SBSConstituentRuntime:
        if not isinstance(control, SBSConstituentControl):
            raise TypeError("control must be SBSConstituentControl")
        return SBSConstituentRuntime(
            profile=self.timing_profile,
            control=control,
            start_ns=start_ns,
            input_frame=PauliFrame() if input_frame is None else input_frame,
            virtual_rotation_provenance=self.virtual_rotation_provenance,
        )

    def run_constituent(
        self,
        control: SBSConstituentControl,
        *,
        start_ns: int = 0,
        input_frame: PauliFrame | None = None,
    ) -> SBSConstituentCycleTrace:
        return self.start_constituent(
            control,
            start_ns=start_ns,
            input_frame=input_frame,
        ).run_to_completion()

    def run_full_xz_cycle(
        self,
        observed_cycle: SBSObservedCycle,
        *,
        start_ns: int = 0,
        input_frame: PauliFrame | None = None,
    ) -> SBSFullXZCycleTrace:
        if not isinstance(observed_cycle, SBSObservedCycle):
            raise TypeError("observed_cycle must be an SBSObservedCycle")
        frame = PauliFrame() if input_frame is None else input_frame
        if not isinstance(frame, PauliFrame):
            raise TypeError("input_frame must be a PauliFrame")
        x_control = SBSConstituentControl(
            quadrature="X",
            observed_class=observed_cycle.syndrome.x,
            reset_action=observed_cycle.reset_actions[0],
            virtual_rotation_key=self.virtual_rotation_key_by_observation[
                observed_cycle.syndrome.x
            ],
        )
        x_trace = self.run_constituent(
            x_control,
            start_ns=start_ns,
            input_frame=frame,
        )
        z_control = SBSConstituentControl(
            quadrature="Z",
            observed_class=observed_cycle.syndrome.z,
            reset_action=observed_cycle.reset_actions[1],
            virtual_rotation_key=self.virtual_rotation_key_by_observation[
                observed_cycle.syndrome.z
            ],
        )
        z_trace = self.run_constituent(
            z_control,
            start_ns=x_trace.end_ns,
            input_frame=x_trace.output_frame,
        )
        return SBSFullXZCycleTrace(
            cycle_index=observed_cycle.cycle_index,
            x_constituent=x_trace,
            z_constituent=z_trace,
            start_ns=start_ns,
            end_ns=z_trace.end_ns,
            total_duration_ns=2 * self.timing_profile.total_duration_ns,
            input_frame=frame,
            output_frame=z_trace.output_frame,
        )
