"""T4.3.3 closed-loop software supervisor for atomic updates and fast fallback.

The supervisor composes the T4.3.2 complete-image bank with the T4.2
bit-accurate fast path.  It never treats a lost commit ack as success, keeps
fast actions defined while the host is absent, and restores last-known-good
contents by publishing them under a new monotonic version rather than rolling
the active version backwards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from cnn_fpga.runtime.atomic_parameter_bank import (
    ActiveImageReadback,
    AtomicParameterBankConfig,
    AtomicParameterBankError,
    AtomicParameterImageBank,
    CommitAck,
    build_parameter_image_manifest,
    serialize_parameter_image,
    verify_commit_ack_readback,
)
from cnn_fpga.runtime.fast_path_fixed_point import (
    BitAccurateFastPath,
    FastPathCodeInput,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTImage


MODEL_SCOPE = "closed_loop_software_fault_recovery_contract_not_rtl_or_board"
PURPOSES = ("candidate", "lkg_republish")
READBACK_STATUSES = (
    "none",
    "confirmed",
    "awaiting_ack_readback",
    "ack_timeout_awaiting_readback",
)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def parameter_image_semantics_sha256(image: ParametricMAPLUTImage) -> str:
    """Hash operational image contents while intentionally excluding version/integrity."""

    if not isinstance(image, ParametricMAPLUTImage):
        raise TypeError("image must be ParametricMAPLUTImage")
    image.verify()
    payload = image.to_dict(include_tables=True)
    for field in ("active_bank_version", "image_crc32", "image_sha256"):
        payload.pop(field)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ClosedLoopRecoveryConfig:
    fast_cycle_ns: int = 5_000
    max_parameter_age_cycles: int = 8_192
    host_timeout_cycles: int = 8_192
    post_commit_guard_cycles: int = 4_000
    guard_blocking_fault_threshold: int = 2
    ack_timeout_cycles: int = 400
    transfer_chunk_bytes: int = 64
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "fast_cycle_ns",
            "max_parameter_age_cycles",
            "host_timeout_cycles",
            "post_commit_guard_cycles",
            "guard_blocking_fault_threshold",
            "ack_timeout_cycles",
            "transfer_chunk_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.max_parameter_age_cycles >= 1 << 16:
            raise ValueError("max parameter age must fit the T4.2 16-bit age word")
        if self.host_timeout_cycles >= 1 << 16:
            raise ValueError("host timeout must fit the trace age word")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")


@dataclass(frozen=True)
class ClosedLoopCycleInput:
    epoch: int
    syndrome_code: int
    syndrome_x: str = "g"
    syndrome_z: str = "g"
    quadrature_phase_bit: int = 0
    ood_score_code: int = 0
    host_heartbeat: bool = False
    communication_available: bool = True
    safe_boundary: bool = True
    reset_ack: bool = False
    observation_valid: bool = True
    input_crc_ok: bool = True
    deadline_ok: bool = True
    reported_integrity_ok: bool = True

    def __post_init__(self) -> None:
        for name in ("epoch", "syndrome_code", "quadrature_phase_bit", "ood_score_code"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.quadrature_phase_bit not in (0, 1):
            raise ValueError("quadrature_phase_bit must be 0 or 1")
        for name in ("syndrome_x", "syndrome_z"):
            if getattr(self, name) not in ("g", "e", "leakage"):
                raise ValueError(f"{name} must be g, e, or leakage")
        for name in (
            "host_heartbeat",
            "communication_available",
            "safe_boundary",
            "reset_ack",
            "observation_valid",
            "input_crc_ok",
            "deadline_ok",
            "reported_integrity_ok",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))


@dataclass(frozen=True)
class UpdateAttempt:
    transaction_id: str
    purpose: str
    accepted: bool
    reason: str
    requested_version: int
    active_version: int
    epoch: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {PURPOSES}")


@dataclass(frozen=True)
class ClosedLoopCycleRecord:
    epoch: int
    active_bank: str
    active_version: int
    active_semantics_sha256: str
    parameter_age_cycles: int
    host_age_cycles: int
    host_timed_out: bool
    communication_available: bool
    commit_status: str
    commit_reason: str
    readback_status: str
    action_mode: str
    action_reason: str
    health_status: str
    reason_trace: str
    active_profile_id: str
    conservative_action: str
    correction_enable: bool
    reset_request: bool
    map_decision_accepted: bool
    fault_flags: tuple[str, ...]
    fault_mask: int
    pauli_frame_x: bool
    pauli_frame_z: bool
    phase_frame_x_code: int
    phase_frame_z_code: int
    recovery_requested: bool
    recovery_reason: str
    guard_blocking_faults: int
    awaiting_readback_version: int | None
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        if self.readback_status not in READBACK_STATUSES:
            raise ValueError(f"readback_status must be one of {READBACK_STATUSES}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fault_flags"] = list(self.fault_flags)
        return payload


class ClosedLoopFaultRecoverySupervisor:
    """Compose atomic bank, ack/readback discipline, guard, LKG and T4.2 fallback."""

    def __init__(
        self,
        images: Sequence[ParametricMAPLUTImage],
        *,
        config: ClosedLoopRecoveryConfig | None = None,
        bank_config: AtomicParameterBankConfig | None = None,
    ) -> None:
        self.config = ClosedLoopRecoveryConfig() if config is None else config
        if not isinstance(self.config, ClosedLoopRecoveryConfig):
            raise TypeError("config must be ClosedLoopRecoveryConfig")
        registered = tuple(images)
        if not registered or not all(isinstance(item, ParametricMAPLUTImage) for item in registered):
            raise ValueError("images must contain ParametricMAPLUTImage records")
        for image in registered:
            image.verify()
        versions = [item.active_bank_version for item in registered]
        if versions[0] != 0 or versions != sorted(versions) or len(set(versions)) != len(versions):
            raise ValueError("images must have unique ascending versions starting at zero")
        if versions != list(range(versions[-1] + 1)):
            raise ValueError("registered image versions must be contiguous")
        if len({item.config for item in registered}) != 1:
            raise ValueError("all images must share one fixed-point config")
        actual_bank = (
            AtomicParameterBankConfig(
                fast_cycle_ns=self.config.fast_cycle_ns,
                max_payload_age_cycles=self.config.max_parameter_age_cycles,
            )
            if bank_config is None
            else bank_config
        )
        if not isinstance(actual_bank, AtomicParameterBankConfig):
            raise TypeError("bank_config must be AtomicParameterBankConfig")
        if actual_bank.fast_cycle_ns != self.config.fast_cycle_ns:
            raise ValueError("supervisor and bank fast-cycle clocks must match")
        if actual_bank.max_payload_age_cycles != self.config.max_parameter_age_cycles:
            raise ValueError("supervisor and bank max payload ages must match")

        self.bank = AtomicParameterImageBank(registered[0], actual_bank)
        self.fast_path = BitAccurateFastPath(
            registered,
            max_parameter_age_cycles=self.config.max_parameter_age_cycles,
        )
        self._images = {item.active_bank_version: item for item in registered}
        self._semantics_by_version = {
            item.active_bank_version: parameter_image_semantics_sha256(item)
            for item in registered
        }
        self._last_cycle = self.fast_path.contract.map_pipeline_cycles - 1
        self._last_host_heartbeat_epoch = self._last_cycle
        self._last_activation_epoch = 0
        self._last_known_good = registered[0]
        self._precommit_lkg: ParametricMAPLUTImage | None = None
        self._candidate_version: int | None = None
        self._candidate_confirmed = False
        self._guard_until_epoch = -1
        self._guard_blocking_faults = 0
        self._recovery_requested = False
        self._recovery_reason = "none"
        self._awaiting_ack: CommitAck | None = None
        self._awaiting_since_epoch: int | None = None
        self._purpose_by_version: dict[int, str] = {}
        self._updates: list[UpdateAttempt] = []
        self._records: list[ClosedLoopCycleRecord] = []

    @property
    def records(self) -> tuple[ClosedLoopCycleRecord, ...]:
        return tuple(self._records)

    @property
    def update_attempts(self) -> tuple[UpdateAttempt, ...]:
        return tuple(self._updates)

    @property
    def recovery_requested(self) -> bool:
        return self._recovery_requested

    @property
    def last_known_good_semantics_sha256(self) -> str:
        return self._semantics_by_version[self._last_known_good.active_bank_version]

    def observe_selection(
        self, *, window_id: int, selection_key: str, eligible: bool
    ) -> None:
        self.bank.observe_selection(
            window_id=window_id, selection_key=selection_key, eligible=eligible
        )

    def submit_update(
        self,
        image: ParametricMAPLUTImage,
        *,
        transaction_id: str,
        selection_key: str,
        source_window_id: int,
        created_epoch: int,
        apply_epoch: int,
        purpose: str = "candidate",
        payload_override: bytes | None = None,
        reverse_chunks: bool = False,
    ) -> UpdateAttempt:
        if purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {PURPOSES}")
        if not isinstance(image, ParametricMAPLUTImage):
            raise TypeError("image must be ParametricMAPLUTImage")
        if image.active_bank_version != self.bank.active_version + 1:
            raise ValueError("image version must be the next active version")
        registered = self._images.get(image.active_bank_version)
        if registered is None or (
            image.image_crc32 != registered.image_crc32
            or image.image_sha256 != registered.image_sha256
        ):
            raise AtomicParameterBankError("unregistered_fast_path_image")
        if purpose == "lkg_republish":
            if not self._recovery_requested:
                raise AtomicParameterBankError("recovery_not_requested")
            if self._semantics_by_version[image.active_bank_version] != self.last_known_good_semantics_sha256:
                raise AtomicParameterBankError("lkg_semantics_mismatch")
        if self._awaiting_ack is not None:
            attempt = UpdateAttempt(
                transaction_id=transaction_id,
                purpose=purpose,
                accepted=False,
                reason="ack_readback_pending",
                requested_version=image.active_bank_version,
                active_version=self.bank.active_version,
                epoch=created_epoch,
                payload_bytes=len(serialize_parameter_image(image)),
            )
            self._updates.append(attempt)
            return attempt
        manifest, payload = build_parameter_image_manifest(
            image,
            transaction_id=transaction_id,
            selection_key=selection_key,
            expected_active_version=self.bank.active_version,
            source_window_id=source_window_id,
            created_epoch=created_epoch,
            apply_epoch=apply_epoch,
            fast_cycle_ns=self.config.fast_cycle_ns,
        )
        transfer_payload = payload if payload_override is None else payload_override
        reason = "staged"
        accepted = False
        try:
            self.bank.begin_stage(manifest, current_epoch=created_epoch)
            size = self.config.transfer_chunk_bytes
            chunks = [
                (offset, transfer_payload[offset : offset + size])
                for offset in range(0, len(transfer_payload), size)
            ]
            if reverse_chunks:
                chunks.reverse()
            for offset, chunk in chunks:
                self.bank.write_chunk(transaction_id, offset=offset, chunk=chunk)
            self.bank.finalize_stage(transaction_id, current_epoch=created_epoch)
            self._purpose_by_version[image.active_bank_version] = purpose
            accepted = True
        except AtomicParameterBankError as exc:
            reason = exc.reason
        attempt = UpdateAttempt(
            transaction_id=transaction_id,
            purpose=purpose,
            accepted=accepted,
            reason=reason,
            requested_version=image.active_bank_version,
            active_version=self.bank.active_version,
            epoch=created_epoch,
            payload_bytes=len(transfer_payload),
        )
        self._updates.append(attempt)
        return attempt

    def submit_lkg_republish(
        self,
        image: ParametricMAPLUTImage,
        *,
        transaction_id: str,
        selection_key: str,
        evidence_window_ids: tuple[int, int],
        created_epoch: int,
        apply_epoch: int,
    ) -> UpdateAttempt:
        first, second = evidence_window_ids
        self.observe_selection(window_id=first, selection_key=selection_key, eligible=True)
        self.observe_selection(window_id=second, selection_key=selection_key, eligible=True)
        return self.submit_update(
            image,
            transaction_id=transaction_id,
            selection_key=selection_key,
            source_window_id=second,
            created_epoch=created_epoch,
            apply_epoch=apply_epoch,
            purpose="lkg_republish",
        )

    def _confirm_readback(self, ack: CommitAck, epoch: int) -> ActiveImageReadback:
        readback = self.bank.readback(epoch=epoch)
        verify_commit_ack_readback(ack, readback)
        purpose = self._purpose_by_version.get(ack.active_version, "candidate")
        self._candidate_confirmed = True
        if purpose == "lkg_republish":
            self._last_known_good = self.bank.read_active_image()
            self._recovery_requested = False
            self._recovery_reason = "none"
            self._guard_blocking_faults = 0
        return readback

    def _process_commit(
        self, cycle: ClosedLoopCycleInput
    ) -> tuple[str, str, str]:
        previous = self.bank.read_active_image()
        ack = self.bank.commit_if_ready(cycle.epoch, safe_boundary=cycle.safe_boundary)
        commit_status = "none" if ack is None else ack.status
        commit_reason = "none" if ack is None else ack.reason
        readback_status = "none"
        if ack is not None and ack.accepted:
            self._precommit_lkg = self._last_known_good
            self._candidate_version = ack.active_version
            self._candidate_confirmed = False
            self._last_activation_epoch = cycle.epoch
            self._guard_until_epoch = cycle.epoch + self.config.post_commit_guard_cycles
            self._guard_blocking_faults = 0
            if cycle.communication_available:
                self._confirm_readback(ack, cycle.epoch)
                readback_status = "confirmed"
            else:
                self._awaiting_ack = ack
                self._awaiting_since_epoch = cycle.epoch
                readback_status = "awaiting_ack_readback"
            if previous.active_bank_version == ack.active_version:
                raise RuntimeError("atomic commit did not change active version")
        elif self._awaiting_ack is not None:
            if cycle.communication_available:
                self._confirm_readback(self._awaiting_ack, cycle.epoch)
                self._awaiting_ack = None
                self._awaiting_since_epoch = None
                readback_status = "confirmed"
            else:
                assert self._awaiting_since_epoch is not None
                age = cycle.epoch - self._awaiting_since_epoch
                readback_status = (
                    "ack_timeout_awaiting_readback"
                    if age > self.config.ack_timeout_cycles
                    else "awaiting_ack_readback"
                )
        return commit_status, commit_reason, readback_status

    def tick(self, cycle: ClosedLoopCycleInput) -> ClosedLoopCycleRecord:
        if not isinstance(cycle, ClosedLoopCycleInput):
            raise TypeError("cycle must be ClosedLoopCycleInput")
        if cycle.epoch != self._last_cycle + 1:
            raise ValueError("epoch must be sequential with no replay or gaps")
        self._last_cycle = cycle.epoch
        if cycle.host_heartbeat and cycle.communication_available:
            self._last_host_heartbeat_epoch = cycle.epoch

        commit_status, commit_reason, readback_status = self._process_commit(cycle)
        image = self.bank.read_active_image()
        parameter_age = min(cycle.epoch - self._last_activation_epoch, (1 << 16) - 1)
        host_age = min(cycle.epoch - self._last_host_heartbeat_epoch, (1 << 16) - 1)
        host_timed_out = host_age > self.config.host_timeout_cycles
        crc = image.image_crc32 if cycle.reported_integrity_ok else "0" * 8
        sha = image.image_sha256 if cycle.reported_integrity_ok else "0" * 64
        result = self.fast_path.step_codes(
            FastPathCodeInput(
                cycle_index=cycle.epoch,
                syndrome_code=cycle.syndrome_code,
                syndrome_x=cycle.syndrome_x,
                syndrome_z=cycle.syndrome_z,
                quadrature_phase_bit=cycle.quadrature_phase_bit,
                expected_active_bank_version=image.active_bank_version,
                reported_image_crc32=crc,
                reported_image_sha256=sha,
                parameter_age_code=parameter_age,
                ood_score_code=cycle.ood_score_code,
                reset_ack=cycle.reset_ack,
                observation_valid=cycle.observation_valid,
                input_crc_ok=cycle.input_crc_ok,
                deadline_ok=cycle.deadline_ok and not host_timed_out,
            )
        )
        action = result.fallback_action
        hardware = action.hardware_action
        blocking = tuple(flag for flag in action.fault_flags if flag != "leakage_observed")
        if self._candidate_version == image.active_bank_version and cycle.epoch <= self._guard_until_epoch:
            if blocking:
                self._guard_blocking_faults += 1
                if self._guard_blocking_faults >= self.config.guard_blocking_fault_threshold:
                    self._recovery_requested = True
                    self._recovery_reason = "post_commit_guard_fault"
        elif (
            self._candidate_version == image.active_bank_version
            and cycle.epoch > self._guard_until_epoch
            and self._candidate_confirmed
            and self._guard_blocking_faults < self.config.guard_blocking_fault_threshold
        ):
            if not self._recovery_requested or self._recovery_reason == "host_timeout_or_parameter_stale":
                self._last_known_good = image
                self._candidate_version = None
                if self._recovery_reason == "host_timeout_or_parameter_stale":
                    self._recovery_requested = False
                    self._recovery_reason = "none"
        if host_timed_out or "parameter_stale" in action.fault_flags:
            self._recovery_requested = True
            self._recovery_reason = "host_timeout_or_parameter_stale"

        awaiting = None if self._awaiting_ack is None else self._awaiting_ack.active_version
        record = ClosedLoopCycleRecord(
            epoch=cycle.epoch,
            active_bank=self.bank.active_bank,
            active_version=image.active_bank_version,
            active_semantics_sha256=self._semantics_by_version[image.active_bank_version],
            parameter_age_cycles=parameter_age,
            host_age_cycles=host_age,
            host_timed_out=host_timed_out,
            communication_available=cycle.communication_available,
            commit_status=commit_status,
            commit_reason=commit_reason,
            readback_status=readback_status,
            action_mode=hardware.mode,
            action_reason=hardware.reason,
            health_status=action.status,
            reason_trace=action.reason_trace,
            active_profile_id=action.active_profile_id,
            conservative_action=action.conservative_action,
            correction_enable=hardware.correction_enable,
            reset_request=hardware.reset_request,
            map_decision_accepted=action.map_decision_accepted,
            fault_flags=action.fault_flags,
            fault_mask=action.fault_mask,
            pauli_frame_x=hardware.pauli_frame_x,
            pauli_frame_z=hardware.pauli_frame_z,
            phase_frame_x_code=hardware.phase_frame_x_code,
            phase_frame_z_code=hardware.phase_frame_z_code,
            recovery_requested=self._recovery_requested,
            recovery_reason=self._recovery_reason,
            guard_blocking_faults=self._guard_blocking_faults,
            awaiting_readback_version=awaiting,
        )
        self._records.append(record)
        return record


__all__ = [
    "MODEL_SCOPE",
    "ClosedLoopRecoveryConfig",
    "ClosedLoopCycleInput",
    "UpdateAttempt",
    "ClosedLoopCycleRecord",
    "ClosedLoopFaultRecoverySupervisor",
    "parameter_image_semantics_sha256",
]
